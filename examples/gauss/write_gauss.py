#*************************************
# Import Statements
#*************************************
import math
import random as rand

#*************************************
# Global Variables
#*************************************
a=1.57866482
b=-0.77648953
c=0.387689533

#*************************************
# Gaussian Function
#*************************************
def gauss(x):
    return a*math.exp(-0.5*(x-b)*(x-b)/(c*c))

xmin=b-5*c
xmax=b+5*c
dx=0.05
nx=int((xmax-xmin)/dx)

# write tranining data
xmin=b-5*c
xmax=b+5*c
f=open("gauss_train.dat","w")
f.write("#X Y\n")
for i in range(0,nx):
    x=xmin+i*dx
    y=gauss(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

# write validation data
xmin=b-3*c
xmax=b+3*c
f=open("gauss_val.dat","w")
f.write("#X Y\n")
for i in range(0,int(nx/5)):
    x=xmin+(xmax-xmin)*rand.random()
    y=gauss(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

# write test data
xmin=b-5*c
xmax=b+5*c
f=open("gauss_test.dat","w")
f.write("#X Y\n")
for i in range(0,nx):
    x=xmin+(xmax-xmin)*rand.random()
    y=gauss(x)
    f.write(str(x)+" "+str(y)+"\n")
f.close()

