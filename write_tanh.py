#*************************************
# Import Statements
#*************************************
import math
import random as rand

#*************************************
# Global Variables
#*************************************
a=1.00

#*************************************
# Tanh Function
#*************************************
def tanh(x):
    return a*math.tanh(x)

xmin=-4.0
xmax=4.0
dx=0.05
nx=int((xmax-xmin)/dx)

# write tranining data
xmin=-4.0
xmax=4.0
f=open("tanh_train.dat","w")
f.write("#X Y\n")
for i in range(0,nx):
    x=xmin+i*dx
    y=tanh(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

# write validation data
xmin=-2.0
xmax=2.0
f=open("tanh_val.dat","w")
for i in range(0,int(nx/5)):
    x=xmin+(xmax-xmin)*rand.random()
    y=tanh(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

# write test data
xmin=-4.0
xmax=4.0
f=open("tanh_test.dat","w")
for i in range(0,nx):
    x=xmin+(xmax-xmin)*rand.random()
    y=tanh(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

