from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# accessing dataset
path = '/Documents/school/CPSC 8420/Final Project/final/NIST'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

# minipulating and splitting dataset
num_classes = len(train[0].unique())
row_num = 8
img_flip = np.transpose(train.values[row_num,1:].reshape(28, 28), axes=[1,0])
trainX, trainy = label(train)
testX, testy = label(test)
k_vals = range(1, 30, 2)
acc = []

for k in range(1, 30, 2):
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainX, trainX)
 	print("The model is trained")
	model.save('mnist_knn_model.h5')
	print("The model is saved")
	score = model.score(trainX, trainy)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	acc.append(score)

i = int(np.argmax(acc))
print("k=%d, Accuracy of %.2f%% " % (k_vals[i],acc[i] * 100))
print(acc)

# test_dataset
model = KNeighborsClassifier(n_neighbors=k_vals[i])
model.fit(trainX, trainy)
predictions = model.predict(testX)
print(predictions)
