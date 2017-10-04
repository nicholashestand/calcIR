
# vars copied from make.inc...I really don't know what I'm doing here
FPIC = -fPIC

NVCCFLAGS = -O3	-DNDEBUG -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function" -DMKL_ILP64
LIB = -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lcublas -lcusparse -lcudart -lcudadevrt
LIB += -lxdrfile
LIB += -lmagma
MKLROOT = /opt/intel/mkl
CUDADIR = /usr/local/cuda
MAGMADIR= /usr/local/magma
LIBDIR  = -L$(CUDADIR)/lib64 \
	  -L$(MKLROOT)/lib/intel64 \
	  -L$(MAGMADIR)/lib

INC     = -I$(CUDADIR)/include \
	  -I$(MKLROOT)/include \
	  -I$(MAGMADIR)/include


all:
	nvcc calcIR.cu -o calcIR.exe $(LIB) $(LIBDIR) $(INC) $(NVCCFLAGS)
