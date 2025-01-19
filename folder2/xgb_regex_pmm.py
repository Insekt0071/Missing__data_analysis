import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import warnings

warnings.filterwarnings('ignore')


def pmm_imputation(data, target_col, predictors, k=5):
    """
    Ulepszona implementacja PMM wykorzystująca XGBoost jako model bazowy
    """
    data_copy = data.copy()

    numeric_predictors = [col for col in predictors
                          if data[col].dtype in ['int64', 'float64']]
    categorical_predictors = [col for col in predictors
                              if data[col].dtype == 'object']

    print(f"PMM for {target_col}:")
    print(f"  Numeric predictors: {numeric_predictors}")
    print(f"  Categorical predictors: {categorical_predictors}")

    # One-hot encoding dla zmiennych kategorycznych
    encoded_data = data_copy.copy()
    for col in categorical_predictors:
        dummies = pd.get_dummies(encoded_data[col], prefix=col, dummy_na=True)
        encoded_data = pd.concat([encoded_data, dummies], axis=1)
        encoded_data.drop(columns=[col], inplace=True)

    # Aktualizacja predyktorów po one-hot encoding
    predictors_encoded = (numeric_predictors +
                          [col for col in encoded_data.columns
                           if any(p in col for p in categorical_predictors)])

    # Podział na obserwacje z i bez braków
    obs_mask = ~data_copy[target_col].isnull()
    donors = encoded_data[obs_mask]
    recipients = encoded_data[~obs_mask]

    if len(recipients) == 0:
        print(f"  No missing values in {target_col}")
        return data

    print(f"  Number of donors: {len(donors)}")
    print(f"  Number of recipients: {len(recipients)}")

    # XGBoost jako model bazowy
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )

    # Trenowanie modelu z early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        donors[predictors_encoded],
        donors[target_col],
        test_size=0.2,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=0
    )

    # Predykcje
    donors_pred = model.predict(donors[predictors_encoded])
    recipients_pred = model.predict(recipients[predictors_encoded])

    # Imputacja z dynamicznym k
    imputed_values = []
    for idx, pred in enumerate(recipients_pred):
        # Dynamiczne k bazujące na gęstości predykcji
        local_k = min(k, int(np.sqrt(len(donors))))

        # Znajdź k najbliższych sąsiadów
        distances = np.abs(donors_pred - pred)
        neighbor_indices = np.argsort(distances)[:local_k]

        # Ważone losowanie bazujące na odległościach
        weights = 1 / (distances[neighbor_indices] + 1e-6)
        weights = weights / weights.sum()

        chosen_idx = np.random.choice(neighbor_indices, p=weights)
        imputed_values.append(donors[target_col].iloc[chosen_idx])

    # Uzupełnienie wartości
    result = data.copy()
    result.loc[~obs_mask, target_col] = imputed_values

    print(f"  Imputation completed for {target_col}")
    return result


def extract_additional_features(df):
    """
    Ekstrakcja dodatkowych cech z tytułów ogłoszeń
    """
    df_copy = df.copy()

    # Lista wzorców do wykrycia
    patterns = {
        'centrum': r'\bcentrum\b',
        'osiedle': r'\bosiedl[ea]\b',
        'nowe': r'\bnow[eya]\b',
        'wyposazenie': r'\bwyposa[zż][oe]n[eya]\b',
        'parking': r'\bparking|garaż|miejsce\s+postojowe\b',
        'balkon': r'\bbalkon|taras\b',
        'metro': r'\bmetr[oa]\b',
        'tramwaj': r'\btramwaj\b',
        'bezposrednio': r'\bbezpośrednio|bez\s+pośredników\b',
    }

    # Tworzenie nowych kolumn
    for name, pattern in patterns.items():
        df_copy[f'title_{name}'] = df_copy['ad_title'].str.lower().str.contains(
            pattern,
            regex=True,
            na=False
        ).astype(int)

    return df_copy


def deduce_from_title(df):
    """
    Dedukuje informacje z tytułów ogłoszeń
    """
    df_copy = df.copy()

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
    """
    Oczyszcza wartości odstające
    """
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
    """
    Przygotowuje dane do modelowania z dodatkowymi cechami
    """
    print("\nStart data preparation...")

    # Ekstrakcja dodatkowych cech
    print("\nExtracting additional features...")
    train = extract_additional_features(train_df)
    test = extract_additional_features(test_df)

    # Imputacja dedukcyjna
    print("\nApplying deductive imputation...")
    train = deduce_from_title(train)
    test = deduce_from_title(test)

    # Czyszczenie outlierów
    print("\nCleaning outliers...")
    train = clean_outliers(train)
    test = clean_outliers(test)

    # Dynamiczny wybór predyktorów
    numeric_predictors = [col for col in train.columns
                          if col.startswith(('flat_', 'title_'))
                          and col not in ['flat_area', 'flat_rooms']
                          and train[col].dtype in ['int64', 'float64']]

    categorical_predictors = ['quarter']

    predictors = numeric_predictors + categorical_predictors
    print("\nSelected predictors for PMM:")
    print("Numeric predictors:", numeric_predictors)
    print("Categorical predictors:", categorical_predictors)

    # Aplikacja PMM dla zmiennych numerycznych
    print("\nApplying PMM imputation...")
    for col in ['flat_area', 'flat_rooms']:
        train = pmm_imputation(train, col, predictors, k=10)
        test = pmm_imputation(test, col, predictors, k=10)

    # Dodanie interakcji między zmiennymi
    print("\nAdding feature interactions...")
    train['area_per_room'] = train['flat_area'] / train['flat_rooms']
    test['area_per_room'] = test['flat_area'] / test['flat_rooms']

    # Usunięcie kolumn, których nie chcemy używać
    print("\nRemoving unnecessary columns...")
    columns_to_drop = ["id", "ad_title", "date_activ", "date_modif", "date_expire"]
    if target_col:
        y_train = train[target_col]
        train = train.drop(columns=columns_to_drop + [target_col])
    test = test.drop(columns=columns_to_drop)

    # Identyfikacja kolumn według typu
    numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train.select_dtypes(include=['object', 'bool']).columns

    print("\nProcessing numeric columns...")
    numeric_data_train = train[numeric_columns]
    numeric_data_test = test[numeric_columns]

    scaler = StandardScaler()
    numeric_scaled_train = pd.DataFrame(
        scaler.fit_transform(numeric_data_train),
        columns=numeric_data_train.columns
    )
    numeric_scaled_test = pd.DataFrame(
        scaler.transform(numeric_data_test),
        columns=numeric_data_test.columns
    )

    print("\nProcessing categorical columns...")
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

    print("\nCombining all features...")
    train_dummies = pd.concat([numeric_scaled_train] + [x[0] for x in categorical_data], axis=1)
    test_dummies = pd.concat([numeric_scaled_test] + [x[1] for x in categorical_data], axis=1)

    common_columns = train_dummies.columns
    test_dummies = test_dummies[common_columns]

    print("\nData preparation completed!")

    if target_col:
        return train_dummies, test_dummies, y_train
    return train_dummies, test_dummies


def train_and_validate_xgboost(X, y, X_test, n_splits=5):
    """
    Trenuje i waliduje model XGBoost z ulepszonymi parametrami
    """
    print("\nStarting XGBoost training and validation...")

    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    scores = {'rmse': [], 'mae': [], 'r2': []}
    feature_importance = pd.DataFrame()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    xgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.005,
        'max_depth': 7,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42
    }

    print("\nXGBoost parameters:")
    for param, value in xgb_params.items():
        print(f"  {param}: {value}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / n_splits

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        scores['rmse'].append(rmse)
        scores['mae'].append(mae)
        scores['r2'].append(r2)

        print(f"Fold {fold} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4.4f}")

        fold_importance = pd.DataFrame({
            'feature': X.columns,
            f'importance_fold_{fold}': model.feature_importances_
        })
        feature_importance = pd.concat([feature_importance, fold_importance], axis=1)

    print("\nOverall CV scores:")
    for metric, values in scores.items():
        mean_score = np.mean(values)
        std_score = np.std(values)
        print(f"{metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")

    feature_importance['importance_mean'] = feature_importance.filter(like='importance').mean(axis=1)
    feature_importance['importance_std'] = feature_importance.filter(like='importance').std(axis=1)
    feature_importance = feature_importance.sort_values('importance_mean', ascending=False)

    return test_predictions, feature_importance, oof_predictions, scores

# Main execution
print("Loading data...")
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

print("\nSaving results...")
# Zapis wyników
output = pd.DataFrame({
    "ID": range(1, len(predictions) + 1),
    "TARGET": predictions
})
output.to_csv("../data/pzn_xgboost_pmm_v3.csv", index=False)

# Zapis feature importance
feature_importance.to_csv("../data/feature_importance_pmm_v3.csv", index=False)

# Zapis metryk CV
cv_metrics = pd.DataFrame(cv_scores)
cv_metrics.to_csv("../data/cv_metrics_pmm_v3.csv", index=False)

print("\nFeature Importance Top 10:")
print(feature_importance[['feature', 'importance_mean', 'importance_std']].head(10))

print("\nProcess completed!")