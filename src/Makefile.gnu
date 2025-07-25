###############################################################################
# COMPILERS AND FLAGS
###############################################################################

CXX_THREAD=mpic++
CXX_SERIAL=g++ 

# production
#CXX_FLAGS=-std=gnu++11 -w -O3 -fno-trapping-math -fno-math-errno -fno-signed-zeros -march=native -DEIGEN_NO_DEBUG 
CXX_FLAGS=-std=gnu++11 -w -O3 -fno-trapping-math -fno-math-errno -fno-signed-zeros -march=native 
# testing
#CXX_FLAGS=-std=gnu++11 -w -O3 -fno-trapping-math -fno-math-errno -fno-signed-zeros -march=native 

