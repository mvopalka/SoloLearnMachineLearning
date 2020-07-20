from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
feature_names = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
X = df[feature_names].values
y = df['Survived'].values

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}

dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

dt = DecisionTreeClassifier(
    max_depth=gs.best_params_['max_depth'],
    min_samples_leaf=gs.best_params_['min_samples_leaf'],
    max_leaf_nodes=gs.best_params_['max_leaf_nodes'])

dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='treeBest', directory='../pictures/DecisionTreeModel', format='png', cleanup=True)
