'''TITANIC SURVIVAL PREDICTION
Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Titanic-Dataset.csv')

df.head()

df.info()

df.describe()

df.duplicated().sum()
#no duplicates found

df.isna().sum()
#there are many null values in cabin

#drop uneeded columns
df.drop(columns=["Cabin",'PassengerId','Name','Ticket'],inplace=True)
#removing null values
df.dropna(inplace=True)


'''Uni Variate Analysis'''

df.head()

'''Survived'''

count = df['Survived'].value_counts()
percent = round(df['Survived'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table

count.plot(kind='bar')
#this shows that more people didnt survive the titanic

'''Passanger Class'''

count = df['Pclass'].value_counts()
percent = round(df['Pclass'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table

count.plot(kind='bar')
#this shows that most people in the titanic where in the 3rd class 

'''Sex'''

count = df['Sex'].value_counts()
percent = round(df['Sex'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table
#this shows that there where more male passengers than female

'''Age'''

Q1 = df.Age.quantile(0.25)
Q3 = df.Age.quantile(0.75)
IQR = Q3 - Q1

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper,lower,sep=' , ')

sns.histplot(data=df , x='Age')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.plot()
#it seems that most passangers where aged between 20-30 years old

sns.boxplot(data=df , x='Age')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.plot()
#there seems to be some outliers after the ages of 65

df[df.Age>=65]

'''Sibling/Spouse'''

count = df['SibSp'].value_counts()
percent = round(df['SibSp'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table

sns.countplot(data=df , x='SibSp')
plt.title('Sibing/Spouse Distribution')
plt.xlabel('sibling/spouse')
plt.ylabel('frequency')
plt.show()
#the majority of passengers didnt come with a sibling or spouse

'''no. of parents/children'''

count = df['Parch'].value_counts()
percent = round(df['Parch'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table

sns.countplot(data=df , x='Parch')
plt.title('Parents and children passengers distribution')
plt.xlabel('paprents/children')
plt.show()
#same as with the spouses it seems that most passengers came alone

'''Fare'''

Q1 = df.Fare.quantile(0.25)
Q3 = df.Fare.quantile(0.75)
IQR = Q3 - Q1

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(upper,lower,sep=' , ')

sns.histplot(data=df , x='Fare')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
#this shows that most passangers have paid between 15-30 

sns.boxplot(data=df , x='Fare')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.show()
#this shows that there are many outliers

df[df.Fare>=100]

'''Embarked'''

count = df['Embarked'].value_counts()
percent = round(df['Embarked'].value_counts(normalize=True)*100,2)

freq_table = pd.DataFrame({'Frequency': count,'Percentage %': percent})
freq_table

sns.countplot(data=df , x='Embarked')
plt.xticks(ticks=['S','C','Q'],labels=['Southampton','Cherbourg','Queenstown'])
plt.title('Embarked distribution')
plt.xlabel('Embarked')
plt.ylabel('Frequency')
plt.show()
#it seems that most pasangers embarked from Southampton

'''MultiVariate Analysis'''

df.groupby('Survived')['Fare'].describe()
#it seems that people who paid more for the titanic survived

df.groupby(['Survived'])['Age'].describe()
#it seems that age doesnt play a factor in who survived or not

df.groupby(['Survived'])['Sex'].value_counts()
#it seems that more males died than females

df.groupby(['Survived'])['SibSp'].value_counts()
#this data shouldn't decide anything as the majority came alone

df.groupby(['Survived'])['Parch'].value_counts()
#this data shouldn't decide anything as the majority came alone

'''Encoding'''

df.head()

model = LabelEncoder()

df['Sex'] = model.fit_transform(df['Sex'])

df['Embarked'] = model.fit_transform(df['Embarked'])

df.head()

'''Features X and target Y'''

X = df.drop(columns=['Survived'])
y = df[['Survived']]

corr_matrix = X.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

'''Data Splitting'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

'''Machine learning models'''

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

print(model.score(X_train,y_train))

print(model.score(X_test,y_test))

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
