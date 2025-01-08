import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Wczytanie danych z folderu 'data'
train_data_path = "../data/pzn-rent-train.csv"
test_data_path = "../data/pzn-rent-test.csv"
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Zastąpienie dziwnych i odstających wartości null'ami
outlier_conditions = {
    "flat_area": lambda x: (x <= 0) | (x > 500),
    "flat_rooms": lambda x: (x <= 0) | (x > 10)
}
for column, condition in outlier_conditions.items():
    train_data.loc[condition(train_data[column]), column] = np.nan
    if column in test_data.columns:
        test_data.loc[condition(test_data[column]), column] = np.nan

# Funkcja do imputacji braków danych
def impute_missing_values(df):
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # Dla kolumn numerycznych
            df[column] = df[column].fillna(df[column].mean())
        elif df[column].dtype == 'bool':  # Dla kolumn logicznych
            df[column] = df[column].fillna(df[column].mode()[0])
        elif df[column].dtype == 'object':  # Dla kolumn kategorycznych
            df[column] = df[column].fillna(df[column].mode()[0])

# Imputacja braków danych
impute_missing_values(train_data)
impute_missing_values(test_data)

# Przygotowanie danych do modelowania
X_train = train_data.drop(columns=["price", "id", "ad_title", "date_activ", "date_modif", "date_expire"])
y_train = train_data["price"]

X_test = test_data.drop(columns=["id", "ad_title", "date_activ", "date_modif", "date_expire"])

# Konwersja wartości logicznych do liczb
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Dopasowanie kolumn w danych testowych do treningowych
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Trenowanie modelu XGBoost
model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Predykcja na danych testowych
predictions = model.predict(X_test)

# Zapis wyników do pliku CSV
output = pd.DataFrame({"ID": range(1, len(predictions) + 1), "TARGET": predictions})
output.to_csv("../data/pzn_xgboost_mean.csv", index=False)

