#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>

void get_eproj( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj);

__global__
void get_eproj_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj);

__global__
void get_kappa_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj, float  *kappa);

__host__ __device__
float minImage( float dx, float boxl );

__host__ __device__
float mag( float dx[3] );

__host__ __device__
float dot3( float x[3], float y[3] );

#endif
