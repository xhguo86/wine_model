from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

d = load_wine()
print(d['DESCR'])
X = pd.DataFrame(d['data'], columns=d['feature_names'])
y = d['target']  # cultivator

def train_model(X, y):
    # create a linear regression model
    model = LogisticRegression()
    model.fit(X, y)
    return model

