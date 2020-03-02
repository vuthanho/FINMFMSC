from numpy.linalg import qr
import numpy as np

X=np.random.rand(5,10)
compressionLevel = 20
r=10
m,n = X.shape
l = min(n,max(compressionLevel,r+10))
OmegaL = np.random.randn(n,l)
Y = np.dot(X,OmegaL)

for _ in range(4):
    Y = qr(Y,mode='reduced')[0]
    S = np.dot(X.T,Y)
    Z = qr(S,mode='reduced')[0]
    Y = np.dot(X,Z)
L = qr(Y,mode='reduced')[0].T