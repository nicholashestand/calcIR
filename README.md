# calcIR

## What is it?
This program calculates Raman and infrared spectroscoy of the OH stretch in bulk liquid water from gromacs molecular dynamics trajectories. The calculation is based on the theory and maps developed by Jim Skinner and coworkers. For more information about the theory, see, for example

Li and Skinner J. Chem. Phys 2010

Gruenbaum et al. JCTC 2013

The references in these papers are also useful. The program is built to run on GPU hardware and is thus very fast.

## How do I use it?
The program can be built using the supplied Makefile. The prerequisits are:
(1) The xdrfile library, which can be downloaded [here](http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library)
(2) CUDA 8.0, which can be downloaded [here](https://developer.nvidia.com/cuda-80-ga2-download-archive)
(3) The Intel MKL math library, which can be downloaded [here](https://software.intel.com/en-us/mkl)
(4) The MAGMA math library for GPU compuatation, version 2.2.0, which can be downloaded [here](http://icl.cs.utk.edu/magma/software/index.html).


The program can be run from the command line using the command

calcIR.exe input.inp

where input.inp is an input file. An example of the necessary components of the input file, along with an explination, is supplied in the input.inp in this repositiory. Once started, the program will read the supplied xtc trajectory file and calculate the IR and Raman spectroscopy of the system.


## Features
This program can calculate infrared and Raman spectroscopy for trajectories obtained using the TIP4P, TIP4P/2005, E3B and E3B3 water models. 

The program will automatically checkpoint its progress after each sample. If for some reason the program crashes, it can be restarted from the checkpoint file using the command

calcIR.exe checkpoint.cpt

where checkpoint.cpt is the name of the checkpoint file. Note that this must end with the extension .cpt.

The makefile compiles a double, with extension '_d' and single precision version of the program. I have found single precision version to be accurate and much faster than the double precision version.

## Contact
This readme is very brief. If questions arise, please contact me at nicholasjhestand@gmail.com.
