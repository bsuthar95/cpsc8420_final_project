# from sklearn.decomposition import PCA
# from keras.datasets import mnist
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# path = '/Documents/school/CPSC 8420/Final Project/final'
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# train.head()


# image = train.iloc[:,1:]
# lbl = train.iloc[:,0:1]
# img = image.values
# img = img.reshape(-1,28,28,1)
# test_img = test.values
# test_img = test_img.reshape(-1,28,28)
# from sklearn.model_selection import train_test_split
# xtrain, xtest, ytrain, ytest = train_test_split(img, lbl, test_size = 0.2, random_state = 1)
# ## Setting the label and the feature columns
# y = train.loc[:,'label'].values
# x = train.loc[:,'pixel0':].values
# pca = PCA(n_components=0.95)
# principalComponents = pca.fit_transform(x)
# pca = PCA().fit(x)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# principal_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 5, c=y, cmap='Spectral')
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('MNIST through PCA', fontsize=24);
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))



path = '/Documents/school/CPSC 8420/Final Project/final'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


image = train.iloc[:,1:]
lbl = train.iloc[:,0:1]
img = image.values
img = img.reshape(-1,28,28,1)
test_img = test.values
test_img = test_img.reshape(-1,28,28)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(img, lbl, test_size = 0.2, random_state = 1)
## Setting the label and the feature columns
y = train.loc[:,'label'].values
x = train.loc[:,'pixel0':].values
pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(x)
pca = PCA().fit(x)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance');
plt.title('PCA MNIST');
plot_digits(x)

