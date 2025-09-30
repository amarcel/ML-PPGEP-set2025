# Importação de dados
import pandas as pd
df = pd.read_excel('dados_estudo_nota.xlsx')
df

# Separação de dados de entrada e de saída:
from sklearn import linear_model
from sklearn import tree

X = df[['horas_estudo']] # Isso é uma matriz (dataframe)
y = df['nota']           # Isso é um vetor (series)

# Criação e treino do modelo:
regressor = linear_model.LinearRegression(fit_intercept=True)
regressor.fit(X, y)

# Cálculo do “a^” e do “b^”:
a, b = regressor.intercept_, regressor.coef_[0]

# Uso do modelo para fazer previsões:
previsao_regressor = regressor.predict(X)

# Cálculo da soma dos erros quadráticos (SEQ):
import numpy as np
seq = np.sum((y - previsao_regressor)**2)

# Mostrando o gráfico de análise:
import matplotlib.pyplot as plt
plt.plot(X['horas_estudo'], y, 'o')
plt.grid(True)
plt.title(f"Relação Horas de estudo vs Nota | SEQ = {seq:.3f}")
plt.xlabel("Horas de estudo")
plt.ylabel("Nota")
plt.plot(X['horas_estudo'], previsao_regressor)
plt.legend(['Observado',
            f'y = {a:.3f} + {b:.3f} x'
            ])
