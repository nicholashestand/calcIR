#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>
#include "magma_v2.h"

__global__
void get_eproj_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, double *eproj);

__global__
void get_kappa_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, double *eproj, 
                    double *kappa, double *mux, double *muy, double *muz);

__global__
void get_spectral_density( double *w, double *MUX, double *MUY, double *MUZ, double *omega, double *Sw, 
                           int nomega, int nchrom, double t1 );

__global__
void cast_to_complex_GPU( double *s_d, magmaDoubleComplex *c_d, int n );
//TODO:: change int to magma_int_t in cast and copy functions

__global__
void copy_complex_GPU( magmaDoubleComplex *out_d, magmaDoubleComplex *in_d, int n );

__host__ __device__
double minImage( double dx, double boxl );

__host__ __device__
double mag3( double dx[3] );

__host__ __device__
double dot3( double x[3], double y[3] );

#endif
