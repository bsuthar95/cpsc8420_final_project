import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam

# define cnn model
def define_model():
	# # model 1
	model = Sequential()
	model.add(Dense(32, activation='relu', input_dim=(28, 28, 1)))
	model.add(Dense(10, activation='softmax'))

	# # model 2
	# model = Sequential()
	# model.add(Dense(128, activation='relu', input_dim=(28, 28, 1)))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(10, activation='softmax'))

	# # model 3
	# model = Sequential()
	# model.add(Dense(32, activation='relu', input_dim=(28, 28, 1)))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(10, activation='softmax'))

	# # model 4
	# model = Sequential()
	# model.add(Dense(128, activation='relu', input_dim=(28, 28, 1)))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(10, activation='softmax'))

	# # model 5
	# model = Sequential()
	# model.add(Dense(32, activation='relu', input_dim=(28, 28, 1)))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dense(10, activation='softmax'))

	# # model 6
	# model = Sequential()
	# model.add(Dense(128, activation='relu', input_dim=(28, 28, 1)))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(128, activation='relu'))	
	# model.add(Dense(128, activation='relu'))
	# model.add(Dense(10, activation='softmax'))


	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

batch_size = 128
num_classes = 10
epochs = 20000
input_shape = (28,28,1)

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
n_folds=5
dataX = trainX
dataY = testX
scores = []
histories = []
# cross validation
kfold = KFold(n_folds, shuffle=True, random_state=1)

# training the CNN model
for train_x, test_x in kfold.split(dataX):
	model = define_model()
	trainX, trainY, testX, testY = dataX[train_x], dataY[train_x], dataX[test_x], dataY[test_x]
	history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)
	print("The model is trained")
	model.save('emnist_dnn_model.h5')
	print("The model is saved")
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	scores.append(acc)
	histories.append(history)

print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
print(scores)