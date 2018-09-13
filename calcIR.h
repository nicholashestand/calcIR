///     Header file for calcIR.cu program       ///

#ifndef CALCIR_H
#define CALCIR_H

// HEADERS

#include <cufft.h>
#include <math.h>
#include "magma_v2.h"
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xdrfile.h>
#include <xdrfile_xtc.h>

// TYPES AND MAGMA REDEFINES

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

// CONSTANTS

#define HBAR        5.308837367       // in cm-1 * ps
#define PI          3.14159265359
#define MAX_STR_LEN 80
#define PSTR        "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PWID        50
#define CP_WRITE    0
#define CP_READ     1
#define CP_INIT     3
#define CHK_ERR     if (Cuerr != cudaSuccess ) { printf(">>> ERROR on CUDA: %s.\n", cudaGetErrorString(Cuerr)); exit(EXIT_FAILURE);}
#define MALLOC_ERR  { printf(">>> ERROR on CPU: out of memory.\n"); exit(EXIT_FAILURE);}
#define CHK_MERR    if (Merr != MAGMA_SUCCESS ) { printf(">>> ERROR on MAGMA: %s.\n", magma_strerror(Merr)); exit(EXIT_FAILURE);}


// FUNCTIONS

__global__
void get_eproj_GPU( rvec *x, float boxx, float boxy, float boxz, int natoms, 
                    int natom_mol, int nchrom, int nchrom_mol, int nmol, int model, user_real_t *eproj);


__global__
void get_kappa_GPU( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                    int nchrom, int nchrom_mol, int nmol, user_real_t *eproj, user_real_t *kappa, 
                    user_real_t *mux, user_real_t *muy, user_real_t *muz, user_real_t *axx,
                    user_real_t *ayy, user_real_t *azz, user_real_t *axy, user_real_t *ayz, 
                    user_real_t *azx, user_real_t avef, int ispecies );

__global__
void get_spectral_density( user_real_t *w, user_real_t *MUX, user_real_t *MUY, user_real_t *MUZ, user_real_t *omega, user_real_t *Sw,
                           int nomega, int nchrom, user_real_t t1, user_real_t avef );


__global__
void cast_to_complex_GPU( user_real_t *s_d, user_complex_t *c_d, magma_int_t n );


__global__
void makeI ( user_complex_t *mat, int n );


__global__
void Pinit ( user_complex_t *prop_d, user_real_t *w_d, int n, user_real_t dt );


__host__ __device__
user_real_t minImage( user_real_t dx, user_real_t boxl );


__host__ __device__
user_real_t mag3( user_real_t dx[3] );


__host__ __device__
user_real_t dot3( user_real_t x[3], user_real_t y[3] );


void ir_init( char *argv[], char gmxf[], char cptf[], char outf[], char model[], int *ifintmeth, user_real_t *dt, int *ntcfpoints, 
              int *nsamples, int *sampleEvery, user_real_t *t1, user_real_t *avef, int *omegaStart, int *omegaStop, 
              int *omegaStep, int *natom_mol, int *nchrom_mol, int *nzeros, user_real_t *beginTime, int *SPECD_FLAG,
              user_real_t *max_int_steps, char species[] );


void printProgress( int currentStep, int totalSteps );


void checkpoint( char *argv[], char gmxf[], char cptf[], char outf[], char model[], int *ifintmeth, user_real_t *dt, int *ntcfpoints, 
                 int *nsamples, int *sampleEvery, user_real_t *t1, user_real_t *avef, int *omegaStart, int *omegaStop, int *omegaStep,
                 int *natom_mol, int *nchrom_mol, int *nzeros, user_real_t *beginTime, int *SPECD_FLAG, user_real_t *max_int_steps, char species[], int nchrom, int nomega, 
                 int *currentSample, int *currentFrame, user_complex_t *tcf, user_complex_t *tcf_vv, user_complex_t *tcf_vh, user_real_t *Sw, 
                 user_complex_t *F_d, user_complex_t *cmux0_d, user_complex_t *cmuy0_d, user_complex_t *cmuz0_d, user_complex_t *caxx0_d, 
                 user_complex_t *cayy0_d, user_complex_t *cazz0_d, user_complex_t *caxy0_d, user_complex_t *cayz0_d, user_complex_t *cazx0_d, int RWI_FLAG );

void signal_handler( int sig );
#endif
