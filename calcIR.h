#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>

void get_eproj( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float *eproj);
float minImage( float dx, float boxl );
float mag( float *dx );
float dot3( float *x, float *y );

#endif
