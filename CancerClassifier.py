#import libraries and data

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split



#Converting the sklearn.dataset cancer to a DataFrame and assigning feature names as columns
#feature_names = (['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       #'mean smoothness', 'mean compactness', 'mean concavity',
       #'mean concave points', 'mean symmetry', 'mean fractal dimension',
       #'radius error', 'texture error', 'perimeter error', 'area error',
       #'smoothness error', 'compactness error', 'concavity error',
       #'concave points error', 'symmetry error', 'fractal dimension error',
       #'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       #'worst smoothness', 'worst compactness', 'worst concavity',
       #'worst concave points', 'worst symmetry', 'worst fractal dimension']
       
cancerdf = pd.DataFrame(cancer.data, columns=[feature_names])

#Assign new column to dataframe
cancerdf = cancerdf.assign(target=pd.DataFrame(cancer.target))

#Assign Dict for target values
dict = {0:'malignant', 1:'benign'}


#Splitting the df into X (the data) and y (the labels) and creating test-train sets
X = cancerdf[cancerdf.columns.drop("target")]
y = cancerdf["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)    

#creating classifiers and setting KNN as "1"
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors = 1)
knn.fit(X_train, y_train)

#estimate the fit of the test set with the training set 
knn.score (X_test, y_test)
                     
#reshape(-1,1) for a single feature; i.e. single column
means = cancerdf.mean()[:-1].values.reshape(1, -1)
knn.predict(means)

cancer_prediction = knn.predict(X_test)


knnscore = knn.score(X_test, y_test)












