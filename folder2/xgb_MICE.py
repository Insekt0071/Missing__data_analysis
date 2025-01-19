import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import multiprocessing as mp
from joblib import Parallel, delayed


class ParallelMICEImputer:
    def __init__(self, n_imputations=5, n_iterations=10, random_state=42, n_jobs=-1):
        self.n_imputations = n_imputations
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.label_encoders = defaultdict(LabelEncoder)
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        np.random.seed(random_state)

    def _initial_imputation(self, data):
        """Perform initial imputation separately for numeric and categorical variables"""
        imputed_data = data.copy()

        for column in data.columns:
            if data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(data[column]):
                    fill_value = data[column].mean()
                    imputed_data[column] = data[column].fillna(fill_value)
                else:
                    fill_value = data[column].mode()[0]
                    imputed_data[column] = data[column].fillna(fill_value)

                    if data[column].dtype.name == 'category':
                        imputed_data[column] = imputed_data[column].astype('category')

        return imputed_data

    def _encode_categorical(self, data, column):
        if not pd.api.types.is_numeric_dtype(data[column]):
            if not self.label_encoders[column].classes_.size:
                self.label_encoders[column].fit(data[column].dropna().unique())
            return pd.Series(self.label_encoders[column].transform(data[column]), index=data.index)
        return data[column]

    def _prepare_features(self, data, target_column):
        X = data.drop(columns=[target_column])
        X_prepared = pd.get_dummies(X, drop_first=True)
        return X_prepared

    def _impute_column(self, data, column):
        missing_mask = data[column].isnull()
        if not missing_mask.any():
            return data[column]

        X = self._prepare_features(data, column)
        y = self._encode_categorical(data, column)

        if pd.api.types.is_numeric_dtype(data[column]):
            model = LinearRegression()
        else:
            model = LogisticRegression(random_state=self.random_state)

        complete_mask = ~missing_mask
        if complete_mask.sum() > 0:
            model.fit(X[complete_mask], y[complete_mask])
            predicted_values = model.predict(X[missing_mask])

            result = y.copy()

            if pd.api.types.is_numeric_dtype(data[column]):
                noise = np.random.normal(0, 0.1 * result[complete_mask].std(), size=missing_mask.sum())
                predicted_values += noise
            else:
                predicted_values = self.label_encoders[column].inverse_transform(predicted_values.astype(int))

            result[missing_mask] = predicted_values
            return result

        return data[column]

    def _perform_single_imputation(self, data, seed):
        np.random.seed(seed)
        imputed_data = self._initial_imputation(data)

        for _ in range(self.n_iterations):
            for col in data.columns:
                if data[col].isnull().any():
                    imputed_data[col] = self._impute_column(imputed_data, col)

        return imputed_data

    def fit_transform(self, data):
        """Perform parallel multiple imputation"""
        seeds = np.random.randint(0, 10000, size=self.n_imputations)

        imputed_datasets = Parallel(n_jobs=self.n_jobs)(
            delayed(self._perform_single_imputation)(data, seed)
            for seed in seeds
        )

        return imputed_datasets


def train_and_predict(train_data, test_data, params):
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']

    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(test_data, drop_first=True)

    # Align columns
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    model = XGBRegressor(**params)
    model.fit(X_train_encoded, y_train)
    return model.predict(X_test_encoded)


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


def prepare_data_for_model(train_df, test_df, target_col=None):
    columns_to_drop = ["id", "ad_title", "date_activ", "date_modif", "date_expire"]
    train = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
    test = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

    if target_col:
        y_train = train[target_col]
        train = train.drop(columns=[target_col])

    train = clean_outliers(train)
    test = clean_outliers(test)

    if target_col:
        return train, test, y_train
    return train, test


# Load and prepare data
train_data = pd.read_csv('../data/pzn-rent-train.csv')
test_data = pd.read_csv('../data/pzn-rent-test.csv')

# Prepare data
X_train, X_test, y_train = prepare_data_for_model(train_data, test_data, 'price')

# Parallel MICE imputation
mice = ParallelMICEImputer(n_imputations=5, n_iterations=10)
imputed_trains = mice.fit_transform(pd.concat([X_train, y_train.to_frame()], axis=1))
imputed_tests = mice.fit_transform(X_test)

# XGBoost parameters - universal settings
xgb_params = {
    'n_estimators': 2000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1  # Will use all available cores
}

# Parallel training and prediction
predictions = Parallel(n_jobs=-1)(
    delayed(train_and_predict)(imputed_trains[i], imputed_tests[i], xgb_params)
    for i in range(len(imputed_trains))
)

# Average predictions
final_predictions = np.mean(predictions, axis=0)

# Save results
output = pd.DataFrame({
    "ID": range(1, len(final_predictions) + 1),
    "TARGET": final_predictions
})
output.to_csv("../data/pzn_xgboost_mice.csv", index=False)