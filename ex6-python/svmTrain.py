from sklearn.svm import SVC

def svmTrain (X, y, C, kernel, gamma='scale'):
	clf = SVC(C = C, kernel=kernel, gamma = gamma, verbose=True)
	return clf.fit (X, y.reshape(-1))

