# Some tips for the assignments in IN3110/IN4110

## Numpy voodoo

```python
import numpy as np


a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

# Resizes the array into a (2,3,3) shape.
# In this case it will copy the first row and extend it, adding a new dimension.
np.resize(a, (2,3,3))

# Result
array([[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]],
       [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]])
```

```python
a = np.array([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]],
             [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]])

# Will sum up the elements on the third axis (Imagine row-wise)
np.sum(a, axis=2)

# Result
array([[ 6, 15, 24],
       [ 6, 15, 24]])
```

```python
a = np.array([[ 6, 15, 24],
       [ 6, 15, 24]])

#Repeat each element 3 times
b = np.repeat(a, 3, axis=1)

# Results into
array([[ 6,  6,  6, 15, 15, 15, 24, 24, 24],
       [ 6,  6,  6, 15, 15, 15, 24, 24, 24]])

# Lets turn it into a different shape
# (using reshape here since we are not extending
#      , but could have also used resize)
b.reshape((2,3,3))

# Result
array([[[ 6,  6,  6],
        [15, 15, 15],
        [24, 24, 24]],

       [[ 6,  6,  6],
        [15, 15, 15],
        [24, 24, 24]]])
```

```python
a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])


# After evaluating 'a % 3', create a boolean array,
# where the values that are 0 are mapped to 'true', elsewhere 'false'
b = a % 3 == 0

array([[False, False,  True],
       [False, False,  True],
       [False, False,  True]])

# Can now use 'b' to index a!
a[b] += 100

# a is now equal to
array([[  1,   2, 103],
       [  4,   5, 106],
       [  7,   8, 109]])

# Also look into np.where
```

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

a[0] # evaluates to [1, 2, 3]

# Some basic slicing
a[:, 0] # [1,4,7] // First element in each row
a[:, 1] # [2,5,6] // Second element in each row
a[:, 2] # [3,6,9] // Third element in each row

# If we have a 3D array
# we can nest the slicing even deeper to get elements from each row like above;)
```

## Numba

```python
from numba import njit, jit, int32

@jit
def fibo(x):
    if x in [0, 1]:
       return 1
    return fibo(x - 1) + fibo(fibo - 2)

# Specifying arguments and return value
# int32 -> int32
@jit(int32(int32))
def fibo(x):
    if x in [0, 1]:
       return 1
    return fibo(x - 1) + fibo(fibo - 2)

# Numba has two compilation modes: nopython mode and object mode.
# The former produces much faster code, but has limitations that can
# force Numba to fall back to the latter.

# To prevent Numba from falling back,
# and instead raise an error, pass nopython=True.
# Quoted from:
# http://numba.pydata.org/numba-doc/0.17.0/user/jit.html#compilation-options (Slightly outdated)
@njit(int32(int32)) # '@njit' is equal to `@jit(nopython=True)`
def fibo(x):
     ...

```

```python
from numba import njit, int64, prange

# Parallelized sum function (https://numba.pydata.org/)
# (array with dtype=int64) -> int64
# Make note that we have to use a different kind of range here 'prange'
@njit(int64(int64[:]), parallel=True)
def parallel_sum(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum += A[i]

    return sum
```
