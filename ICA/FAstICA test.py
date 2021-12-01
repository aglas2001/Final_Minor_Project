# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:32:00 2021

@author: aglas
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
#import csv

X, y = load_digits(return_X_y=True)
a,b = X.shape

transformer = FastICA(n_components=25,random_state=0, max_iter = 200, tol =2e-5)
X_transformed = transformer.fit_transform(X)
print(a,b)
c,d = X_transformed.shape
print(c,d)








#f = open('C:/Users/aglas/OneDrive/Bureaublad/Documenten/GitHub/Final_Minor_Project/ICA/csv_test.csv','w')
#writer =csv.writer(f)

#a,b = X.shape
#for i in range(a):
#    writer.writerow(X[i])
#f.close()

#f = open('C:/Users/aglas/OneDrive/Bureaublad/Documenten/GitHub/Final_Minor_Project/ICA/csv_test2.csv','w')
#writer =csv.writer(f)

#for i in range(c):
#    writer.writerow(X_transformed[i])
#f.close()
