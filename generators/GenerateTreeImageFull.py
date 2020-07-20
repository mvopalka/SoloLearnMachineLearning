from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
feature_names = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
# dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)

dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='treeFull', directory='../pictures/DecisionTreeModel', format='png', cleanup=True)
