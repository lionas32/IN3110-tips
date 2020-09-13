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
