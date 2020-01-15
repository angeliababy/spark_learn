
import numpy as np
import time

w = 1000
A = np.array(range(w**2)).reshape(w, w)
B = np.array(range(w**2)).reshape(w, w)
time1 = time.time()
C=np.dot(A,B)
#C = A*B
print(time.time()-time1)
print("A:\n%s" % A)
print("B:\n%s" % B)
print("C:\n%s" % C)
