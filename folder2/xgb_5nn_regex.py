import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import warnings

warnings.filterwarnings('ignore')


def deduce_from_title(df):
    df_copy = df.copy()

    # Funkcja do wyciągania liczb z tekstu
    def extract_first_number(text):
        if not isinstance(text, str):
            return None
        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None

    # 1. Imputacja flat_area
    area_pattern = r'\b(\d+)\s*m[²2\s]'
    mask_area = df_copy['flat_area'].isnull()
    for idx in df_copy[mask_area].index:
        title = df_copy.loc[idx, 'ad_title']
        if isinstance(title, str):
            match = re.search(area_pattern, title)
            if match:
                df_copy.loc[idx, 'flat_area'] = float(match.group(1))

    # 2 & 3. Imputacja flat_rooms
    rooms_patterns = [
        (r'\b(\d+)[-\s]?(?:pok|pomieszcz|pokojowe|pokoi)\w*', lambda x: float(x)),
        (r'\bkawalerk[ai]\b', lambda x: 1.0)
    ]

    mask_rooms = df_copy['flat_rooms'].isnull()
    for idx in df_copy[mask_rooms].index:
        title = df_copy.loc[idx, 'ad_title']
        if isinstance(title, str):
            for pattern, transform in rooms_patterns:
                match = re.search(pattern, title.lower())
                if match:
                    value = transform(match.group(1)) if match.groups() else transform(None)
                    df_copy.loc[idx, 'flat_rooms'] = value
                    break

    # 4. Imputacja flat_furnished
    furnished_pattern = r'\bumeblowane?\b'
    mask_furnished = df_copy['flat_furnished'].isnull()
    for idx in df_copy[mask_furnished].index:
        title = df_copy.loc[idx, 'ad_title']
        if isinstance(title, str) and re.search(furnished_pattern, title.lower()):
            df_copy.loc[idx, 'flat_furnished'] = True

    return df_copy


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
    # Najpierw wykonujemy imputację dedukcyjną
    train = deduce_from_title(train_df)
    test = deduce_from_title(test_df)

    # Czyszczenie outlierów
    train = clean_outliers(train)
    test = clean_outliers(test)

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
        all_categories = pd.concat([train[col], test[col]]).unique()
        dummies_train = pd.get_dummies(train[col], prefix=col, dummy_na=False)
        dummies_test = pd.get_dummies(test[col], prefix=col, dummy_na=False)

        missing_cols_train = set(dummies_test.columns) - set(dummies_train.columns)
        missing_cols_test = set(dummies_train.columns) - set(dummies_test.columns)

        for c in missing_cols_train:
            dummies_train[c] = 0
        for c in missing_cols_test:
            dummies_test[c] = 0

        categorical_data.append((dummies_train, dummies_test))

    train_dummies = pd.concat([numeric_final_train] + [x[0] for x in categorical_data], axis=1)
    test_dummies = pd.concat([numeric_final_test] + [x[1] for x in categorical_data], axis=1)

    common_columns = train_dummies.columns
    test_dummies = test_dummies[common_columns]

    if target_col:
        return train_dummies, test_dummies, y_train
    return train_dummies, test_dummies


def train_and_validate_xgboost(X, y, X_test, n_splits=5):
    # Inicjalizacja list na wyniki
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    scores = {'rmse': [], 'mae': [], 'r2': []}
    feature_importance = pd.DataFrame()

    # Konfiguracja walidacji krzyżowej
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Parametry modelu
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # szybsza metoda trenowania
        'random_state': 42
    }

    # Trenowanie modeli
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")

        # Podział danych
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Inicjalizacja i trenowanie modelu
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        # Predykcje
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        # Zapisanie predykcji OOF
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / n_splits

        # Obliczenie metryk
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        scores['rmse'].append(rmse)
        scores['mae'].append(mae)
        scores['r2'].append(r2)

        print(f"Fold {fold} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

        # Feature importance
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            f'importance_fold_{fold}': model.feature_importances_
        })
        feature_importance = pd.concat([feature_importance, fold_importance], axis=1)

    # Podsumowanie wyników
    print("\nOverall CV scores:")
    for metric, values in scores.items():
        mean_score = np.mean(values)
        std_score = np.std(values)
        print(f"{metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")

    # Agregacja feature importance
    feature_importance['importance_mean'] = feature_importance.filter(like='importance').mean(axis=1)
    feature_importance['importance_std'] = feature_importance.filter(like='importance').std(axis=1)
    feature_importance = feature_importance.sort_values('importance_mean', ascending=False)

    return test_predictions, feature_importance, oof_predictions, scores


# Wczytanie danych
train_data = pd.read_csv('../data/pzn-rent-train.csv')
test_data = pd.read_csv('../data/pzn-rent-test.csv')

# Przygotowanie danych
X_train, X_test, y_train = prepare_data_with_aligned_categories(
    train_data,
    test_data,
    target_col='price'
)

# Trenowanie modelu z walidacją krzyżową
predictions, feature_importance, oof_predictions, cv_scores = train_and_validate_xgboost(
    X_train, y_train, X_test, n_splits=5
)

# Zapis wyników
output = pd.DataFrame({"ID": range(1, len(predictions) + 1), "TARGET": predictions})
output.to_csv("../data/pzn_xgboost_5nn_deductive_cv.csv", index=False)

# Zapis feature importance
feature_importance.to_csv("../data/feature_importance.csv", index=False)

# Zapis metryk CV
cv_metrics = pd.DataFrame(cv_scores)
cv_metrics.to_csv("../data/cv_metrics.csv", index=False)

print("\nFeature Importance Top 10:")
print(feature_importance[['feature', 'importance_mean', 'importance_std']].head(10))