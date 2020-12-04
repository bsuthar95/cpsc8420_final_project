from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# accessing dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainY = to_categorical(trainY)
testY = to_categorical(testY)

k_vals = range(1, 30, 2)
acc = []

for k in range(1, 30, 2):
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainX, trainX)
 	print("The model is trained")
	model.save('mnist_knn_model.h5')
	print("The model is saved")
	score = model.score(trainX, trainY)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	acc.append(score)

i = int(np.argmax(acc))
print("k=%d, Accuracy of %.2f%% " % (k_vals[i],acc[i] * 100))
print(acc)

# test_dataset
model = KNeighborsClassifier(n_neighbors=k_vals[i])
model.fit(trainX, trainY)
predictions = model.predict(testX)
print(predictions)
