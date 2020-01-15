import numpy as np 
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='cpu')
def vectorAdd(a, b):
    return a + b

def main():
    N = 320000000

    A = np.ones(N, dtype=np.float32 )
    B = np.ones(N, dtype=np.float32 )
    C = np.zeros(N, dtype=np.float32 )

    start = timer()
    C = vectorAdd(A, B)
    vectorAdd_time = timer() - start

    print("c[:5] = " + str(C[:5]))
    print("c[-5:] = " + str(C[-5:]))

    print("vectorAdd took %f seconds " % vectorAdd_time)

if __name__ == '__main__':
    main()
