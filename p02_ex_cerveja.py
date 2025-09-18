# Importação dos dados:
import pandas as pd

df = pd.read_excel("dados_cerveja.xlsx")
df.head()

# Separação de dados de entrada (X) e de saída (y) do modelo:
entradas = ['temperatura','copo','espuma','cor']
saida = 'classe'
X = df[entradas]
y = df[saida]
X = X.replace({
    "mug": 1, "pint":2,
    "sim":1, "não":0,
    "clara": 0, "escura":1,
})

# Criação e treino do modelo:
from sklearn import tree
arvore = tree.DecisionTreeClassifier()
arvore.fit(X, y)

# Uso do modelo para fazer predição:
arvore.predict([[-5, 2, 1, 0]])

# Exibição do desenho da árvore de decisão:
import matplotlib.pyplot as plt
plt.figure(dpi=400)

tree.plot_tree(arvore,
               feature_names=entradas,
               class_names=arvore.classes_,
               filled=True)

# Exibição das probabilidades de predição de cada classe:
proba = arvore.predict_proba([[-1, 1, 0, 0]])[0]
pd.Series(proba, index=arvore.classes_)
