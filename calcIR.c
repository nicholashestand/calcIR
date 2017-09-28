/*This is my first attempt to port my python ir program to cuda. 
 * It currently suffers from very slow excecution in python. 
 * I'm going to try to port it to cuda c */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xdrfile/xdrfile.h>
#include <xdrfile/xdrfile_xtc.h>
#include "calcIR.h"


int main()
{
    // Files to read from. TODO: make as arguments or read from input file
    char *gmxf          = "./traj_comp.xtc";
    char *ndxf          = "./index.ndx";
    int  frame0         = 0;
    int  framelast      = 1;

    // Variables
    int i;
    int natoms, nmol, nchrom;
    int frame;
    int natom_mol = 4; // change depending on water molecule
    int nchrom_mol = 2; // chromophores per molecule (2 for stretch)

    // Trajectory stuff
    rvec    *x;
    float  **kappa;
    matrix box;
    float  boxl;
    float time, prec;
    int step;

    // electric field
    float *eproj;


    printf("Will read the trajectory from: %s.\n",gmxf);

    // Open the trajectory file
    XDRFILE *trj = xdrfile_open( gmxf, "r" );

    // get the number of atoms and molecules
    read_xtc_natoms( gmxf, &natoms);
    nmol = natoms / natom_mol;
    nchrom = nmol * nchrom_mol;
    printf("Found %d atoms and %d molecules.\n",natoms, nmol);
    printf("Found %d chromophores.\n",nchrom);


    // allocate memory arrays
    x       = calloc(natoms, sizeof(x[0]));         // position vector matrix
    eproj   = malloc(nchrom*sizeof(float  *));      // electric field vector matrix
    kappa   = malloc(nchrom*sizeof(float  *));      // hamiltonian matrix
    for (i=0; i<nchrom; i++){
        kappa[i] = malloc(nchrom*sizeof(double));
    }


    // loop over the trajectory
    for ( frame=frame0; frame<framelast; frame++ ){
        // read the trajectory
        read_xtc( trj, natoms, &step, &time, box, x, &prec );
        // assume isotropic box assign length
        boxl = box[0][0];

        get_eproj( x, boxl, natoms, natom_mol, nchrom, nchrom_mol, nmol, eproj );//TODO run on GPU
        exit(0);

    }

    free(x);
    free(kappa);
    free(eproj);
}

// BUILD ELECTRIC FIELD PROJECTION
// TODO: Move these to external
// MOVE TO GPU
//
void get_eproj( rvec *x, float boxl, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, float *eproj )
{
    
    int n, m, i, j;
    int chrom, atom;
    float mox[DIM];                     // oxygen position on molecule m
    float mx[DIM];                      // atom position on molecule m
    float nhx[DIM];                     // hydrogen position on molecule n of the current chromophore
    float nox[DIM];                     // oxygen position on molecule n
    float nohx[DIM];                    // the unit vector pointing along the OH bond for the current chromophore
    float dr[DIM];                      // the min image vector between two atoms
    float r;                            // the distance between two atoms 
    const float cutoff = 0.7831;        // the oh cutoff distance
    const float bohr_nm = 18.8973;      // convert from bohr to nanometer
    rvec efield[nchrom];                // the electric field

    // This outer loop can be moved to the gpu I think
    for ( chrom = 0; chrom < nchrom; chrom++){ 

        // define the molecule index of the current chromophore
        n = chrom / nchrom_mol;

        // initialize the electric field vector to zero at this chromophore
        for (j=0; j<DIM; j++){
            efield[chrom][j] = 0.;
        }

        // get the position of the current hydrogen corresponding to the current chromophore
        // NOTE: I'm making some assumptions about the ordering of the positions, this can be changed if necessary
        if ( chrom % 2 == 0 ){ //HW1
            nhx[0] = x[ n*natom_mol + 1 ][0];
            nhx[1] = x[ n*natom_mol + 1 ][1];
            nhx[2] = x[ n*natom_mol + 1 ][2];
        }
        else if ( chrom % 2 == 1 ){ //HW2
            nhx[0] = x[ n*natom_mol + 2 ][0];
            nhx[1] = x[ n*natom_mol + 2 ][1];
            nhx[2] = x[ n*natom_mol + 2 ][2];
        }

        // The oxygen position
        nox[0] = x[ n*natom_mol ][0];
        nox[1] = x[ n*natom_mol ][1];
        nox[2] = x[ n*natom_mol ][2];

        // The oh unit vector
        nohx[0]  = minImage( nhx[0] - nox[0], boxl );
        nohx[1]  = minImage( nhx[1] - nox[1], boxl );
        nohx[2]  = minImage( nhx[2] - nox[2], boxl );
        r        = mag(nohx);
        nohx[0]  /= r;
        nohx[1]  /= r;
        nohx[2]  /= r;
        // test is good...
        //printf("muhat: %f %f %f\n", nohx[0], nohx[1], nohx[2]);


        // Loop over all other molecules
        for ( m = 0; m < nmol; m++ ){

            // skip if the atom belongs to the molecule on the current chromophore
            if ( m == n ) continue;

            // get oxygen position on current molecule m
            // Im assuming the oxygen molecule is the first in the position list, but this can be changed if necessary
            // For now, I'm just trying to get up and running quickly
            mox[0] = x[ m*natom_mol ][0];
            mox[1] = x[ m*natom_mol ][1];
            mox[2] = x[ m*natom_mol ][2];

            // find displacement between oxygen on m and hydrogen on n
            dr[0]  = minImage( mox[0] - nhx[0], boxl );
            dr[1]  = minImage( mox[1] - nhx[1], boxl );
            dr[2]  = minImage( mox[2] - nhx[2], boxl );
            r      = mag(dr);

            // skip if the distance is greater than the cutoff
            if ( r > cutoff ) continue;

            // loop over all atoms in the current molecule and calculate the electric field (excluding the oxygen atoms! since they have no charge)
            for ( i=1; i < natom_mol; i++ ){
                // TODO: clean this up...
                mx[0] = x[ m*natom_mol + i ][0];
                mx[1] = x[ m*natom_mol + i ][1];
                mx[2] = x[ m*natom_mol + i ][2];

                // the displacement betweent the h on n and the current atom on m, in bohr so the efield will be in au
                dr[0]  = minImage( nhx[0] - mx[0], boxl )*bohr_nm;
                dr[1]  = minImage( nhx[1] - mx[1], boxl )*bohr_nm;
                dr[2]  = minImage( nhx[2] - mx[2], boxl )*bohr_nm;
                r      = mag(dr);
 
                if ( i < 3  ){ // the hydrogens TODO:: MAKE THIS MORE USER FRIENDLY
                    for ( j=0; j < DIM; j++){
                        efield[chrom][j] += 0.52 * dr[j] / (r*r*r);
                    }
                }
                else if ( i == 3 ){ // The M sites
                    for ( j=0; j < DIM; j++){
                        efield[chrom][j] -= 1.04 * dr[j] / (r*r*r);
                    }
                }
            } // end loop over atoms in molecule m

        } // end loop over molecules m

    // project the efield along the OH bond to get the relevant value for the map
    eproj[chrom] = dot3( efield[chrom], nohx );
    // test looks good, everything appears to be ok
    //printf("chrom: %d, eproj %f \n", chrom, eproj[chrom]);

    } // end loop over chromophores
}








// RANDOM HELPER FUNCTIONS
// TODO MAKE So it can be called on gpu from make efield calc
float minImage( float dx, float boxl )
{
    return dx - boxl*round(dx/boxl);
}

float mag( float * dx )
{
    return sqrt( dot3( dx, dx ) );
}

float dot3( float *x, float *y )
{
    return  x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}
