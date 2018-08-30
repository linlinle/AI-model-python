"""
在整个 iris 数据集上训练的上述树的 graphviz 导出示例; 其结果被保存在 iris.pdf 中
"""

from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data,iris.target)

# 经过训练，我们可以使用 export_graphviz 导出器以 Graphviz 格式导出决策树.

#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("iris")

#   func:`export_graphviz` 出导出还支持各种美化，包括通过他们的类着色节点
dot_data = tree.export_graphviz(clf, out_file=None,
                            feature_names=iris.feature_names,
                            class_names=iris.target_names,
                            filled=True, rounded=True,
                            special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")