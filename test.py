#!/usr/bin/env python3
import numpy as np

# from scipy.stats import multivariate_normal
# mean = np.array([0.5, 0.1, 0.3])
# cov = np.array([[0.1, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.9]])
# x = np.random.uniform(size=(100, 3))
# y = multivariate_normal.pdf(x, mean=mean, cov=cov)

X=np.random.randn(3,5)
Y=X[X>0]
print(X)
print(Y)

