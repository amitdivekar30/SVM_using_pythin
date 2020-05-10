# SVM
# classify the Size_Categorie using SVM

# month	month of the year: 'jan' to 'dec'
# day	day of the week: 'mon' to 'sun'
# FFMC	FFMC index from the FWI system: 18.7 to 96.20
# DMC	DMC index from the FWI system: 1.1 to 291.3
# DC	DC index from the FWI system: 7.9 to 860.6
# ISI	ISI index from the FWI system: 0.0 to 56.10
# temp	temperature in Celsius degrees: 2.2 to 33.30
# RH	relative humidity in %: 15.0 to 100
# wind	wind speed in km/h: 0.40 to 9.40
# rain	outside rain in mm/m2 : 0.0 to 6.4
# Size_Categorie 	the burned area of the forest ( Small , Large)

import pandas as pd 
import numpy as np 
import seaborn as sns

dataset = pd.read_csv("forestfires.csv")
dataset.head()
dataset.describe()
dataset.columns

sns.boxplot(x="size_category",y="temp",data=dataset,palette = "hls")
sns.boxplot(x="wind",y="size_category",data=dataset,palette = "hls")
sns.pairplot(data=dataset)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X = dataset.iloc[:, 2:30].values
y = dataset.iloc[:, [30]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(X_train, y_train)
pred_test_linear = model_linear.predict(X_test)

np.mean(pred_test_linear==y_test) # Accuracy = 62.26

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(X_train, y_train)
pred_test_poly = model_poly.predict(X_test)

np.mean(pred_test_poly==y_test) # Accuracy = 68.55

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train, y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==y_test) # Accuracy = 68.22




