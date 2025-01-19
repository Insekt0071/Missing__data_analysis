import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def clean_outliers(df):
    df_clean = df.copy()
    outlier_conditions = {
        "flat_area": lambda x: (x <= 0) | (x > 500),
        "flat_rooms": lambda x: (x <= 0) | (x > 10)
    }
    for column, condition in outlier_conditions.items():
        if column in df.columns:
            df_clean.loc[condition(df_clean[column]), column] = np.nan
    return df_clean


def prepare_data_with_aligned_categories(train_df, test_df, target_col=None):
    # Utworzenie kopii dataframe'ów
    train = train_df.copy()
    test = test_df.copy()

    # Usunięcie kolumn, których nie chcemy używać
    columns_to_drop = ["id", "ad_title", "date_activ", "date_modif", "date_expire"]
    if target_col:
        y_train = train[target_col]
        train = train.drop(columns=columns_to_drop + [target_col])
    test = test.drop(columns=columns_to_drop)

    # Identyfikacja kolumn według typu
    numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train.select_dtypes(include=['object', 'bool']).columns

    # Obsługa zmiennych numerycznych
    numeric_data_train = train[numeric_columns]
    numeric_data_test = test[numeric_columns]

    # Standardyzacja danych numerycznych
    scaler = StandardScaler()
    numeric_scaled_train = pd.DataFrame(
        scaler.fit_transform(numeric_data_train),
        columns=numeric_data_train.columns
    )
    numeric_scaled_test = pd.DataFrame(
        scaler.transform(numeric_data_test),
        columns=numeric_data_test.columns
    )

    # Imputacja danych numerycznych
    imputer = KNNImputer(n_neighbors=5)
    numeric_imputed_train = pd.DataFrame(
        imputer.fit_transform(numeric_scaled_train),
        columns=numeric_scaled_train.columns
    )
    numeric_imputed_test = pd.DataFrame(
        imputer.transform(numeric_scaled_test),
        columns=numeric_scaled_test.columns
    )

    # Przywrócenie oryginalnej skali
    numeric_final_train = pd.DataFrame(
        scaler.inverse_transform(numeric_imputed_train),
        columns=numeric_imputed_train.columns
    )
    numeric_final_test = pd.DataFrame(
        scaler.inverse_transform(numeric_imputed_test),
        columns=numeric_imputed_test.columns
    )

    # Obsługa zmiennych kategorycznych
    categorical_data = []
    for col in categorical_columns:
        # Połączenie unikalnych wartości z obu zbiorów
        all_categories = pd.concat([train[col], test[col]]).unique()

        # Utworzenie dummy variables dla obu zbiorów z tymi samymi kategoriami
        dummies_train = pd.get_dummies(train[col], prefix=col, dummy_na=False)
        dummies_test = pd.get_dummies(test[col], prefix=col, dummy_na=False)

        # Wyrównanie kolumn między zbiorami
        missing_cols_train = set(dummies_test.columns) - set(dummies_train.columns)
        missing_cols_test = set(dummies_train.columns) - set(dummies_test.columns)

        for c in missing_cols_train:
            dummies_train[c] = 0
        for c in missing_cols_test:
            dummies_test[c] = 0

        categorical_data.append((dummies_train, dummies_test))

    # Połączenie wszystkich zmiennych
    train_dummies = pd.concat([numeric_final_train] + [x[0] for x in categorical_data], axis=1)
    test_dummies = pd.concat([numeric_final_test] + [x[1] for x in categorical_data], axis=1)

    # Upewnienie się, że kolejność kolumn jest taka sama
    common_columns = train_dummies.columns
    test_dummies = test_dummies[common_columns]

    if target_col:
        return train_dummies, test_dummies, y_train
    return train_dummies, test_dummies


# Wczytanie danych
train_data = pd.read_csv('../data/pzn-rent-train.csv')
test_data = pd.read_csv('../data/pzn-rent-test.csv')

# Przygotowanie danych z poprawną obsługą kategorii
X_train, X_test, y_train = prepare_data_with_aligned_categories(
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
output.to_csv("../data/pzn_xgboost_5nn.csv", index=False)