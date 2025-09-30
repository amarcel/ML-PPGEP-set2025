# Importação de dados
import pandas as pd
df = pd.read_excel('dados_estudo_nota.xlsx')
df

# Separação de dados de entrada e de saída:
from sklearn.tree import DecisionTreeRegressor

X = df[['horas_estudo']] # Isso é uma matriz (dataframe)
y = df['nota']           # Isso é um vetor (series)

# Criação e treino do modelo:
tree_reg = DecisionTreeRegressor(random_state=42,
                                 max_depth=3)
tree_reg.fit(X, y)

# Uso do modelo para fazer previsões:
previsao_tree_reg = tree_reg.predict(X)

# Cálculo da soma dos erros quadráticos (SEQ):
import numpy as np
seq = np.sum((y - previsao_tree_reg)**2)


# Mostrando o gráfico de análise:
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Nota')
ax.set_title(f'Relação entre Horas de Estudo e Nota com Estimativa de Árvore de Decisão (max_depth=3) | SEQ = {seq:.2f}') 
ax.grid(True)
ax.plot(X, y, 'o', label='Dados')
ax.plot(X, previsao_tree_reg, 'r-', label='Estimativa Árvore de Decisão')
ax.legend()
plt.show()
