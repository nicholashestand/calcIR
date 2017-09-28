#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>

void get_eproj( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float *eproj);
__global__
void get_eproj_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float *eproj);
__host__ __device__
float minImage( float dx, float boxl );
__host__ __device__
float mag( float *dx );
__host__ __device__
float dot3( float *x, float *y );

#endif
