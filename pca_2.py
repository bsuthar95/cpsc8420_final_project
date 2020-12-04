from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# accessing dataset
path = '/Documents/school/CPSC 8420/Final Project/final'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()

# minipulating and splitting dataset
image = train.iloc[:,1:]
lbl = train.iloc[:,0:1]
img = image.values
img = img.reshape(-1,28,28,1)
test_img = test.values
test_img = test_img.reshape(-1,28,28)
xtrain, xtest, ytrain, ytest = train_test_split(img, lbl, test_size = 0.2, random_state = 1)
y = train.loc[:,'label'].values
x = train.loc[:,'pixel0':].values
pca = PCA(n_components=0.95)
pri_components = pca.fit_transform(x)
pca = PCA().fit(x)

principal_df = pd.DataFrame(data = pri_components, columns = ['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# plotting the dataset
plt.scatter(pri_components[:, 0], pri_components[:, 1], s= 5, c=y, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('MNIST through PCA', fontsize=24);
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

