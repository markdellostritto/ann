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
def gauss(x,y):
    return a*math.exp(-0.5*((x-b)*(x-b)+(y-b)*(y-b))/(c*c))

xmin=b-3*c
xmax=b+3*c
ymin=b-3*c
ymax=b+3*c
dx=0.05
dy=0.05
nx=int((xmax-xmin)/dx)
ny=int((ymax-ymin)/dy)

print 'nx = ',nx
print 'ny = ',ny

# write tranining data
xmin=b-3*c
xmax=b+3*c
ymin=b-3*c
ymax=b+3*c
f=open("gauss_train.dat","w")
f.write("#X Y Z\n")
for i in range(0,nx):
    for j in range(0,ny):
        x=xmin+i*dx
        y=ymin+j*dy
        z=gauss(x,y)
        f.write(str(x)+" "+str(y)+" "+str(z)+"\n")
f.close()

# write validation data
xmin=b-1*c
xmax=b+1*c
ymin=b-1*c
ymax=b+1*c
f=open("gauss_val.dat","w")
f.write("#X Y Z\n")
for i in range(0,int(nx/3)):
    for j in range(9,int(ny/3)):
        x=xmin+(xmax-xmin)*rand.random()
        y=ymin+(ymax-ymin)*rand.random()
        z=gauss(x,y)
        f.write(str(x)+" "+str(y)+" "+str(z)+"\n")
f.close()

# write test data
xmin=b-3*c
xmax=b+3*c
ymin=b-3*c
ymax=b+3*c
f=open("gauss_test.dat","w")
f.write("#X Y Z\n")
for i in range(0,nx):
    for j in range(0,ny):
        x=xmin+(xmax-xmin)*rand.random()
        y=ymin+(ymax-ymin)*rand.random()
        z=gauss(x,y)
        f.write(str(x)+" "+str(y)+" "+str(z)+"\n")
f.close()

