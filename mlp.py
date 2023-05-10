from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


#X, y = make_classification(n_samples=100, random_state=1)
#print(type(X))
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
#                                                    random_state=1)

#print(X_train[0])
#clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
#clf.predict_proba(X_test[:1])
#
#clf.predict(X_test[:5, :])
#
#clf.score(X_test, y_test)

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)