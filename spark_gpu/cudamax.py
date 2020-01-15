
from numba import cuda
import numpy as np

@cuda.jit
def max_example(result, values):
    """Find the maximum value in values and store in result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    print(tid,bid,bdim)
    i = (bid * bdim) + tid
    cuda.atomic.max(result, 0, values[i])


arr = np.random.rand(16384)
result = np.zeros(1, dtype=np.float64)

max_example[256,64](result, arr)
print(result[0]) # Found using cuda.atomic.max
print(max(arr))  # Print max(arr) for comparision (should be equal!)

@cuda.jit
def max_example_3d(result, values):
    """
    Find the maximum value in values and store in result[0].
    Both result and values are 3d arrays.
    """
    i, j, k = cuda.gridprint(3)
    # Atomically store to result[0,1,2] from values[i, j, k]
    cuda.atomic.max(result, (0, 1, 2), values[i, j, k])

arr = np.random.rand(1000).reshape(10,10,10)
result = np.zeros((3, 3, 3), dtype=np.float64)
max_example_3d[(2, 2, 2), (5, 5, 5)](result, arr)
print(result[0, 1, 2], '==', np.max(arr))

