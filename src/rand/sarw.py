# **********************************
# import statements
# **********************************
import numpy as np
import random as rng

# **********************************
# constants
# **********************************
ll=1.5
N=5

# **********************************
# array
# **********************************
arr=np.zeros(3*N).reshape(N,3)

# **********************************
# generate random walk
# **********************************
for i in range(1,N):
    # select previous point in walk
    v=np.copy(arr[i-1])
    # check if v is in array (i.e. collision)
    inlist=np.any(np.all(v == arr, axis=1))
    # while collision==true, generate a new point
    while(inlist):
        # pick a random coordinate (x, y, or z)
        j=rng.randint(0,2)
        # generate random number in [0,1]
        r=rng.random()
        # generate a new point on the walk
        v=np.copy(arr[i-1])
        if(r<0.5): v[j]=v[j]-1
        else: v[j]=v[j]+1
        # check if v is in array (i.e. collision)
        inlist=np.any(np.all(v == arr, axis=1))
    # if no collision, append to the walk
    arr[i]=np.copy(v)

# **********************************
# print random walk
# **********************************
for a in arr: print(a)

