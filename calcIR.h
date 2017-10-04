#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>

void get_eproj( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj);

// a convience function to request eigenspectrum from magma
void diagonalize_kappa(float *kappa, float *w, int n, int frame);

__global__
void get_eproj_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj);

__global__
void get_kappa_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float  *eproj, 
                    float  *kappa, float *mux, float *muy, float *muz);

__global__
void get_spectral_density( float *w, float *MUX, float *MUY, float *MUZ, float *omega, float *Sw, 
                           int nomega, int nchrom, float t1 );

__host__ __device__
float minImage( float dx, float boxl );

__host__ __device__
float mag3( float dx[3] );

__host__ __device__
float dot3( float x[3], float y[3] );

#endif
