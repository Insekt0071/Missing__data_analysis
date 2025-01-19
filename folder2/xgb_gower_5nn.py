import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def gower_distance(X, Y=None):
    """
    Implementacja odległości Gowera dla różnych typów zmiennych.

    Parameters:
    -----------
    X : DataFrame
        Pierwsza macierz obserwacji
    Y : DataFrame, optional
        Druga macierz obserwacji (jeśli None, oblicza odległości w X)

    Returns:
    --------
    distances : ndarray
        Macierz odległości Gowera
    """
    if Y is None:
        Y = X

    # Zachowanie nazw kolumn i ich typów
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    binary_cols = X.select_dtypes(include=['bool']).columns

    # Inicjalizacja macierzy odległości
    gower_mat = np.zeros((X.shape[0], Y.shape[0]))
    n_vars = len(numeric_cols) + len(categorical_cols) + len(binary_cols)

    # Obliczanie odległości dla zmiennych numerycznych
    for col in numeric_cols:
        # Normalizacja zakresu
        range_col = np.ptp(np.concatenate([X[col], Y[col]]))
        if range_col != 0:
            dist_mat = cdist(
                X[col].values.reshape(-1, 1) / range_col,
                Y[col].values.reshape(-1, 1) / range_col,
                metric='cityblock'
            )
            gower_mat += dist_mat

    # Obliczanie odległości dla zmiennych kategorycznych
    for col in categorical_cols:
        dist_mat = (X[col].values.reshape(-1, 1) != Y[col].values.reshape(1, -1)).astype(float)
        gower_mat += dist_mat

    # Obliczanie odległości dla zmiennych binarnych
    for col in binary_cols:
        dist_mat = (X[col].values.reshape(-1, 1) != Y[col].values.reshape(1, -1)).astype(float)
        gower_mat += dist_mat

    return gower_mat / n_vars


def knn_impute_with_gower(X, k=5):
    """
    Imputacja metodą k najbliższych sąsiadów z wykorzystaniem odległości Gowera.
    """
    X_imputed = X.copy()

    # Dla każdej kolumny z brakami
    for col in X.columns:
        # Znajdujemy wiersze z brakami w danej kolumnie
        missing_mask = X[col].isna()
        if not missing_mask.any():
            continue

        # Dzielimy dane na kompletne i niekompletne
        X_missing = X[missing_mask]
        X_complete = X[~missing_mask]

        if X_complete.empty:
            continue

        # Obliczamy odległości Gowera
        distances = gower_distance(X_missing, X_complete)

        # Dla każdego wiersza z brakami znajdujemy k najbliższych sąsiadów
        for i, row_idx in enumerate(np.where(missing_mask)[0]):
            # Znajdujemy indeksy k najbliższych sąsiadów
            neighbor_idx = np.argsort(distances[i])[:k]

            # Obliczamy wartość do imputacji (średnia dla num, moda dla kat)
            if X[col].dtype in ['int64', 'float64']:
                imputed_value = X_complete.iloc[neighbor_idx][col].mean()
            else:
                imputed_value = X_complete.iloc[neighbor_idx][col].mode()[0]

            X_imputed.iloc[row_idx, X_imputed.columns.get_loc(col)] = imputed_value

    return X_imputed


def prepare_data_with_gower(train_df, test_df, target_col=None):
    """
    Przygotowanie danych z imputacją wykorzystującą odległość Gowera.
    """
    # Utworzenie kopii dataframe'ów
    train = train_df.copy()
    test = test_df.copy()

    # Usunięcie niepotrzebnych kolumn
    columns_to_drop = ["id", "ad_title", "date_activ", "date_modif", "date_expire"]
    if target_col:
        y_train = train[target_col]
        train = train.drop(columns=columns_to_drop + [target_col])
    test = test.drop(columns=columns_to_drop)

    # Czyszczenie outlierów
    outlier_conditions = {
        "flat_area": lambda x: (x <= 0) | (x > 500),
        "flat_rooms": lambda x: (x <= 0) | (x > 10)
    }
    for column, condition in outlier_conditions.items():
        train.loc[condition(train[column]), column] = np.nan
        if column in test.columns:
            test.loc[condition(test[column]), column] = np.nan

    # Imputacja z wykorzystaniem odległości Gowera
    train_imputed = knn_impute_with_gower(train, k=5)
    test_imputed = knn_impute_with_gower(test, k=5)

    # Konwersja zmiennych kategorycznych na dummy variables
    train_dummies = pd.get_dummies(train_imputed, drop_first=True)
    test_dummies = pd.get_dummies(test_imputed, drop_first=True)

    # Wyrównanie kolumn
    for col in train_dummies.columns:
        if col not in test_dummies.columns:
            test_dummies[col] = 0
    for col in test_dummies.columns:
        if col not in train_dummies.columns:
            train_dummies[col] = 0

    # Zapewnienie tej samej kolejności kolumn
    common_columns = train_dummies.columns
    test_dummies = test_dummies[common_columns]

    if target_col:
        return train_dummies, test_dummies, y_train
    return train_dummies, test_dummies


# Wczytanie danych
train_data = pd.read_csv('../data/pzn-rent-train.csv')
test_data = pd.read_csv('../data/pzn-rent-test.csv')

# Przygotowanie danych z wykorzystaniem odległości Gowera
X_train, X_test, y_train = prepare_data_with_gower(
    train_data,
    test_data,
    target_col='price'
)

# Parametry XGBoost
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# Trenowanie modelu
model = XGBRegressor(**xgb_params)
model.fit(X_train, y_train)

# Predykcja
predictions = model.predict(X_test)

# Zapis wyników
output = pd.DataFrame({"ID": range(1, len(predictions) + 1), "TARGET": predictions})
output.to_csv("../data/pzn_xgboost_gower.csv", index=False)