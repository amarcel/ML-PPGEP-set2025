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
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X, y)

# Uso do modelo para fazer previsões:
previsao_tree_reg = tree_reg.predict(X)

# Cálculo da soma dos erros quadráticos (SEQ):
seq = np.sum((df['nota'] - previsao_tree_reg)**2)

# Mostrando o gráfico de análise:
fig, ax = plt.subplots()
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Nota')
ax.set_title(f'Relação entre Horas de Estudo e Nota com Estimativa de Árvore de Decisão (max_depth=3) | SEQ = {seq:.2f}') 
ax.grid(True)
ax.plot(horas_estudo, nota, 'o', label='Dados')
ax.plot(X_plot.flatten(), y_plot, 'r-', label='Estimativa Árvore de Decisão')
ax.legend()
plt.show()
