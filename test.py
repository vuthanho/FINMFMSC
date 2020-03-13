#!/usr/bin/env python3
import numpy as np

# from scipy.stats import multivariate_normal
# mean = np.array([0.5, 0.1, 0.3])
# cov = np.array([[0.1, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.9]])
# x = np.random.uniform(size=(100, 3))
# y = multivariate_normal.pdf(x, mean=mean, cov=cov)

# def f():
#     X=np.random.randn(3,5)
#     Y=X[X>0]
#     return {'test' : Y}
# res = f()
# print(res['test'])

# X = np.random.randn(3,6)
# GradX = np.random.randn(3,6)
# print(X)
# print(GradX)
# print()
# print(X>0)
# print()
# print(GradX<0)
# print()
# print(np.any(np.dstack((X>0,GradX<0)),2))

Res = {}
Res.update({"emwnenmf" : {}})
Res.update({"incal" : []})
for i in range(10):
    Res['emwnenmf'].update({i : i})
m = 'emwnenmf'
print(Res[m][9])

Res['incal'].append(np.arange(10))
print(Res['incal'][0].min())