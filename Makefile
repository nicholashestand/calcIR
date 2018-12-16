src     = calcIR.cu
exes    = calcIR.exe
NVCC    = nvcc
INC     = -I$(CUDADIR)/include -I$(MKLROOT)/include -I$(MAGMADIR)/include
FLAGS	= -Xcompiler "-fPIC -Wall -Wno-unused-function" -DMKL_ILP64 -Wno-deprecated-gpu-targets
LIBS    = -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lcufft -lmagma -lxdrfile
LIBDIRS = -L/opt/intel/mkl/lib/intel64 -L/usr/local/cuda-8.0/lib64 -L/usr/local/magma/lib
INCDIRS = -I/opt/intel/mkl/include -I/usr/local/cuda-8.0/include -I/usr/local/magma/include


all: ${exes}

${exes}: ${src}
	$(NVCC) $(src) -o $(exes) $(FLAGS) $(LIBDIRS) $(LIBS) $(INCDIRS)

clean:
	rm calcIR.exe
