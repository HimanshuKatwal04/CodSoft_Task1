''' IRIS FLOWER CLASSIFICATION'''

'''Use the Iris dataset to develop a model that can classify iris flowers into different species based on their sepal and petal measurements. This dataset is widely used for introductory classification tasks.'''

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use("fivethirtyeight")
%matplotlib inline

df=pd.read_csv('/kaggle/input/iris-codsoft/IRIS.csv')
df.head()

#information about the dataset
df.info()

#describing about the dataset
df.describe()

df.shape

df.head()

#count the value
df['species'].value_counts()

#finding the null value
df.isnull().sum()

import missingno as msno
msno.bar(df)

df.drop_duplicates(inplace=True)

# EDA

#1. Relationship between species and sepal length

plt.figure(figsize=(15,8))
sns.boxplot(x='species',y='sepal_length',data=df.sort_values('sepal_length',ascending=False))

#2. Relationship between species and sepal width

df.plot(kind='scatter',x='sepal_width',y='sepal_length')

#3. Relationship between sepal width and sepal length

sns.jointplot(x="sepal_length", y="sepal_width", data=df, size=5)

#4.Pairplot

sns.pairplot(df, hue="species", size=3)

#5. Boxplot

df.boxplot(by="species", figsize=(12, 6))

#5. Andrews_curves

import pandas.plotting
from pandas.plotting import andrews_curves
andrews_curves(df, "species")

#6.CategoricalPlot

plt.figure(figsize=(15,15))
sns.catplot(x='species',y='sepal_width',data=df.sort_values('sepal_width',ascending=False),kind='boxen')

#7.Violinplot

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_width',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=df)

# Neural Network

X=df.drop('species',axis=1)
y=df['species']

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df['species'] = pd.Categorical(df.species)
df['species'] = df.species.cat.codes
# Turn response variable into one-hot response vectory = to_categorical(df.response)
y =to_categorical(df.species)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y,random_state=123)

model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(4,)))

model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=45,validation_data=(X_test, y_test))

model.evaluate(X_test,y_test)

pred = model.predict(X_test[:10])
print(pred)

p=np.argmax(pred,axis=1)
print(p)
print(y_test[:10])

history.history['accuracy']

history.history['val_accuracy']

plt.figure()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()
