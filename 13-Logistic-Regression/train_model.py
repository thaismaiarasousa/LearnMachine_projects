# pylint: disable=import-error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler

# Carregar os dados
ad_data = pd.read_csv('C:/Users/thais/Desktop/LearnMachine/13-Logistic-Regression/advertising.csv')

# Selecionar recursos e destino
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Normalizar os dados usando StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Criar e ajustar o modelo de regressão logística com mais iterações
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train_scaled, y_train)

# Salvar o modelo treinado
joblib.dump(logmodel, 'logistic_regression_model.pkl')
