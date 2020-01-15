
from numbapro import jit, int32, float32, complex64

@jit(complex64(int32, float32, complex64), target="cpu")
def bar(a, b, c):
   return a + b  * c

@jit(complex64(int32, float32, complex64)) # target kwarg defaults to "cpu"
def foo(a, b, c):
   return a + b  * c

time1 = time.time()
print(foo)
print("cpu:%f",time.time()-time1)
time1 = time.time()
print(foo(1, 2.0, 3.0j))
print("gpu:%f",time.time()-time1)
