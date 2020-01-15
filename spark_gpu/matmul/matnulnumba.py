
import numpy as np
from numba import guvectorize,cuda 
import time
from numba import *

@cuda.jit
def matnulgpu(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        print(C.shape[0],C.shape[1])
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
            #print(i,j,k)
        
        C[i, j] = tmp

@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'],
             '(m,n),(n,p)->(m,p)',target="cuda")
def matnulgpu1(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'],
             '(m,n),(n,p)->(m,p)',target="cpu")
def matnulcpu(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


w = 5
A = np.array(range(w**2)).reshape(w, w)
B = np.array(range(w**2)).reshape(w, w)
time1 = time.time()
print(A,B)
C = matnulcpu(A, B)
print("cpu:%f",time.time()-time1)
print("C:\n%s" % C)
time1 = time.time()
#A=cuda.to_device(A)
#B=cuda.to_device(B)
C = np.zeros((w, w), dtype = np.uint8)
blockdim = (32, 32)
griddim = ((w+32-1)//32,(w+32-1)//32)

matnulgpu[griddim, blockdim](A,B,C) 
#C = matmulgpu(A, B)
#A=A.copy_to_host()
#B=B.copy_to_host()
#C=C.copy_to_host()
print("gpu1:%f",time.time()-time1)
print("C:\n%s" % C)
time1 = time.time()
C = matnulgpu1(A, B)
print("gpu2:%f",time.time()-time1)
print("C:\n%s" % C)
time1 = time.time()
#fast_matmul[griddim, blockdim](A,B,C)
#print("gpu3:%f",time.time()-time1)
#print("C:\n%s" % C)

#print("A:\n%s" % A)
#print("B:\n%s" % B)
#print("C:\n%s" % C)

