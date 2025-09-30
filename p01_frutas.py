# Importação dos dados:
import pandas as pd

df = pd.read_excel("dados_frutas.xlsx")
df

# Separação de dados de entrada (X) e de saída (y) do modelo:
y = df['Fruta']
caracteristicas = ["Arredondada","Suculenta",'Vermelha','Doce']
X = df[caracteristicas]

# Criação e treino do modelo:
from sklearn import tree
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# Uso do modelo para fazer predição:
arvore.predict([[0, 0, 0, 0]])

# Exibição do desenho da árvore de decisão:
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(dpi=400, figsize=[4,4])

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)
plt.show()

# Exibição das probabilidades de predição de cada classe:
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
