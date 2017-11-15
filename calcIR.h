#ifndef CALCIR_H
#define CALCIR_H

#include <xdrfile/xdrfile.h>
#include "magma_v2.h"

#ifdef USE_DOUBLES

typedef double user_real_t;
typedef magmaDoubleComplex user_complex_t;

#define MAGMA_ONE  MAGMA_Z_ONE
#define MAGMA_ZERO MAGMA_Z_ZERO
#define MAGMA_MAKE MAGMA_Z_MAKE
#define MAGMA_ADD  MAGMA_Z_ADD
#define MAGMA_MUL  MAGMA_Z_MUL
#define MAGMA_DIV  MAGMA_Z_DIV
#define MAGMA_REAL MAGMA_Z_REAL
#define MAGMA_IMAG MAGMA_Z_IMAG

#else

typedef float user_real_t;
typedef magmaFloatComplex user_complex_t;

#define MAGMA_ONE  MAGMA_C_ONE
#define MAGMA_ZERO MAGMA_C_ZERO
#define MAGMA_MAKE MAGMA_C_MAKE
#define MAGMA_ADD  MAGMA_C_ADD
#define MAGMA_MUL  MAGMA_C_MUL
#define MAGMA_DIV  MAGMA_C_DIV
#define MAGMA_REAL MAGMA_C_REAL
#define MAGMA_IMAG MAGMA_C_IMAG

#endif

#define HBAR 5.308837367       // in cm-1 * ps
#define PI   3.14159265359
#define MAX_STR_LEN 80
#define PSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PWID 50
#define CP_W 0
#define CP_R 1

// FUNCTIONS

__global__
void get_eproj_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, int model, user_real_t *eproj);


__global__
void get_kappa_GPU( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, user_real_t *eproj,
                    user_real_t *kappa, user_real_t *mux, user_real_t *muy, user_real_t *muz, user_real_t avef);

__global__
void get_spectral_density( user_real_t *w, user_real_t *MUX, user_real_t *MUY, user_real_t *MUZ, user_real_t *omega, user_real_t *Sw,
                           int nomega, int nchrom, user_real_t t1, user_real_t avef );


__global__
void cast_to_complex_GPU( user_real_t *s_d, user_complex_t *c_d, magma_int_t n );


__host__ __device__
user_real_t minImage( user_real_t dx, user_real_t boxl );


__host__ __device__
user_real_t mag3( user_real_t dx[3] );


__host__ __device__
user_real_t dot3( user_real_t x[3], user_real_t y[3] );


void ir_init( char *argv[], char gmxf[], char cptf[], char outf[], char model[], int *ifintmeth, user_real_t *dt, int *ntcfpoints, 
              int *nsamples, int *sampleEvery, user_real_t *t1, user_real_t *avef, int *omegaStart, int *omegaStop, 
              int *omegaStep, int *natom_mol, int *nchrom_mol, int *nzeros, user_real_t *beginTime, int *ispecd,
              user_real_t *max_int_steps);


void printProgress( int currentStep, int totalSteps );

void checkpoint( char cptf[], int *currentSample, int *currentFrame, user_complex_t *tcf, int ntcfpoints, user_complex_t *F, 
                 int nchrom2, user_complex_t *cmux0, user_complex_t *cmuy0, user_complex_t *cmuz0, int nchrom, 
                 int ispecd, user_real_t *Sw, user_real_t *omega, int nomega, int RW_FLAG );

#endif
