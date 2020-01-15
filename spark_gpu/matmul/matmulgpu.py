
import numpy as np
from numba import guvectorize
import time

@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'],
             '(m,n),(n,p)->(m,p)',target="cuda")
def matmulgpu(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'],
             '(m,n),(n,p)->(m,p)',target="cpu")
def matmulcpu(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


w = 2000
A = np.array(range(w**2)).reshape(w, w)
B = np.array(range(w**2)).reshape(w, w)
time1 = time.time()
C = matmulcpu(A, B)
print("cpu:%f",time.time()-time1)
time1 = time.time()
C = matmulgpu(A, B)
print("gpu:%f",time.time()-time1)
print("A:\n%s" % A)
print("B:\n%s" % B)
print("C:\n%s" % C)
