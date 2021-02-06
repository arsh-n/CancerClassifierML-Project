#import libraries and data

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


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















