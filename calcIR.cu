/*  This program calculates the OH stetch IR absorption spectrum
 *  for coupled water from an MD trajectory. The exciton Hamilt-
 *  onian is built using the maps developed by Skinner  and  co-
 *  workers
 */

#include "calcIR.h" 

int main(int argc, char *argv[])
{


    // Some help for starting the program. User must supply a single argument
    if ( argc != 2 ){
        printf("Usage:\n"
               "\tInclude as the first argument the name of an input file. No other arguments are allowed.\n");
        exit(EXIT_FAILURE);   
    }


    // retrieve and print info about gpu
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("\nGPU INFO:\n"
           "\tDevice name: %s\n"
           "\tMemory: %g gb\n",
           prop.name, prop.totalGlobalMem/(1.E9));
    

    // ***              Variable Declaration            *** //
    // **************************************************** //

    printf("\n>>> Setting default parameters\n");

    // Model parameters
    char          gmxf[MAX_STR_LEN]; strncpy( gmxf, "traj.xtc", MAX_STR_LEN );   // trajectory file
    char          outf[MAX_STR_LEN]; strncpy( outf, "spec", MAX_STR_LEN );   // name for output files
    char          cptf[MAX_STR_LEN]; strncpy( cptf, "spec", MAX_STR_LEN );   // name for output files
    char          model[MAX_STR_LEN];strncpy( model,"e3b3", MAX_STR_LEN );   // water model tip4p, tip4p2005, e3b2, e3b3
    int           imodel        = 0;                                      // integer for water model
    int           imap          = 0;                                      // integer for spectroscopic map used (0 - 2013 Gruenbaum) (1 - 2010 Li)
    int           ispecies      = 0;                                      // integer for species of interest
    int           ntcfpoints    = 150 ;                                   // the number of tcf points for each spectrum
    int           nsamples      = 1   ;                                   // number of samples to average for the total spectrum
    float         sampleEvery   = 10. ;                                   // sample a new configuration every sampleEvery ps. Note the way the program is written, 
                                                                          // ntcfpoints*dt must be less than sampleEvery.
    user_real_t   omegaStart    = 2000;                                   // starting frequency for spectral density
    user_real_t   omegaStop     = 5000;                                   // ending frequency for spectral density
    int           omegaStep     = 5;                                      // resolution for spectral density
    int           natom_mol     = 4;                                      // Atoms per water molecule  :: MODEL DEPENDENT
    int           nchrom_mol    = 2;                                      // Chromophores per molecule :: TWO for stretch -- ONE for bend
    int           nzeros        = 25600;                                  // zeros for padding fft

    user_real_t   dt            = 0.010;                                  // dt between frames in xtc file (in ps)
    user_real_t   beginTime     = 0   ;                                   // the beginning time in ps to allow for equilibration, if desired
    user_real_t   t1            = 0.260;                                  // relaxation time ( in ps )
    user_real_t   avef          = 3415.2;                                 // the approximate average stretch frequency to get rid of high
                                                                          // frequency oscillations in the time correlation function
    char          species[MAX_STR_LEN]; strncpy( species, " ", MAX_STR_LEN ); // species HOD/H2O HOD/D2O H2O D2O

 
    // read in model parameters
    // START FROM INPUT FILE
    ir_init( argv, gmxf, cptf, outf, model, &dt, &ntcfpoints, &nsamples, &sampleEvery, &t1, 
            &avef, &omegaStart, &omegaStop, &omegaStep, &natom_mol, &nchrom_mol, &nzeros, &beginTime,
            species, &imap );


    // Print the parameters to stdout
    printf("\tSetting xtc file %s\n",                       gmxf        );
    printf("\tSetting default file name to %s\n",           outf        );
    printf("\tSetting model to %s\n",                       model       );
    printf("\tSetting the number of tcf points to %d\n",    ntcfpoints  );
    printf("\tSetting nsamples to %d\n",                    nsamples    ); 
    printf("\tSetting sampleEvery to %f (ps)\n",            sampleEvery );
    printf("\tSetting omegaStep to %d\n",                   omegaStep   );
    printf("\tSetting natom_mol to %d\n",                   natom_mol   );
    printf("\tSetting nchrom_mol to %d\n",                  nchrom_mol  );
    printf("\tSetting nzeros to %d\n",                      nzeros      );
    printf("\tSetting map to %d\n",                         imap        );
    printf("\tSetting species to %s\n",                     species     );
    printf("\tSetting omegaStart to %f\n",                  omegaStart  );
    printf("\tSetting omegaStop to %f\n",                   omegaStop   );
    printf("\tSetting dt to %f\n",                          dt          );
    printf("\tSetting t1 to %f (ps)\n",                     t1          );
    printf("\tSetting avef to %f\n",                        avef        );
    printf("\tSetting equilibration time to %f (ps)\n",     beginTime   );

    // Useful variables and condstants
    int                 natoms, nmol, nchrom;                                           // number of atoms, molecules, chromophores
    int                 currentSample   = 0;                                            // current sample
    int                 currentFrame    = 0;                                            // current frame
    const int           ntcfpointsR     = ( nzeros + ntcfpoints - 1 ) * 2;              // number of points for the real fourier transform
    const int           nomega          = ( omegaStop - omegaStart ) / omegaStep + 1;   // number of frequencies for the spectral density
    magma_int_t         nchrom2;                                                        // nchrom squared
    float               desired_time;                                                   // desired time for the current frame
    int                 nframes, est_nframes;                                           // variables for indexing offsets


    // Trajectory variables for the CPU
    rvec                *x;                                                             // Position vector
    matrix              box;                                                            // Box vectors
    float               gmxtime, prec;                                                  // Time at current frame, precision of xtf file
    int                 step, xdrinfo;                                                  // The current step number
    int64_t             *frame_offset;                                                  // Offset for random frame access from trajectory
    float               frame_dt;                                                       // Time between successive frames

    // GPU variables                 
    const int           blockSize = 128;                                                // The number of threads to launch per block
    rvec                *x_d;                                                           // positions
    user_real_t         *mux_d,   *muy_d,   *muz_d;                                     // transition dipole moments
    user_real_t         *axx_d,   *ayy_d,   *azz_d;                                     // polarizability
    user_real_t         *axy_d,   *ayz_d,   *azx_d;                                     // polarizability
    user_complex_t      *cmux0_d, *cmuy0_d, *cmuz0_d;                                   // complex version of the transition dipole moment at t=0 
    user_complex_t      *cmux_d,  *cmuy_d,  *cmuz_d;                                    // complex version of the transition dipole moment
    user_complex_t      *caxx0_d, *cayy0_d, *cazz0_d;                                   // complex version of the polarizability at t=0
    user_complex_t      *caxy0_d, *cayz0_d, *cazx0_d;                                   // complex version of the polarizability at t=0
    user_complex_t      *caxx_d,  *cayy_d,  *cazz_d;                                    // complex version of the polarizability
    user_complex_t      *caxy_d,  *cayz_d,  *cazx_d;                                    // complex version of the polarizability
    user_complex_t      *tmpmu_d;                                                       // to sum all polarizations
    user_real_t         *MUX_d, *MUY_d, *MUZ_d;                                         // transition dipole moments in the eigen basis
    user_real_t         *eproj_d;                                                       // the electric field projected along the oh bonds
    user_real_t         *kappa_d;                                                       // the hamiltonian on the GPU
    user_real_t         *kappa;

    // magma variables for ssyevd
    user_real_t         aux_work[1];                                                    // To get optimal size of lwork
    magma_int_t         aux_iwork[1], info;                                             // To get optimal liwork, and return info
    magma_int_t         lwork, liwork;                                                  // Leading dim of kappa, sizes of work arrays
    magma_int_t         *iwork;                                                         // Work array
    user_real_t         *work;                                                          // Work array
    user_real_t         *w   ;                                                          // Eigenvalues
    user_real_t         wi   ;                                                          // Eigenvalues
    user_real_t         *wA  ;                                                          // Work array
    int                 SSYEVD_ALLOC_FLAG = 1;                                          // flag whether to allocate ssyevr arrays -- it is turned off after they are allocated


    // magma variables for gemv
    magma_queue_t       queue;

    // variables for spectrum calculations
    user_real_t         *w_d;                                                           // Eigenvalues on the GPU
    user_real_t         *omega, *omega_d;                                               // Frequencies on CPU and GPU
    user_real_t         *Sw, *Sw_d;                                                     // Spectral density on CPU and GPU
    user_real_t         *tmpSw;                                                         // Temporary spectral density
    user_real_t         *Rw;                                                            // inverse participation ratio weighted frequency distribution 
    user_real_t         *Rmw;                                                            // inverse participation ratio weighted frequency distribution 
    user_real_t         *Pw;                                                            // frequency distribution
    user_real_t         ipr;                                                            // inverse participation ratio
    user_real_t         mipr;                                                           // molecular inverse participation ratio



    // variables for TCF
    user_complex_t      *F_d;                                                           // F matrix on GPU
    user_complex_t      *prop_d;                                                        // Propigator matrix on GPU
    user_complex_t      *ctmpmat_d;                                                     // temporary complex matrix for matrix multiplications on gpu
    user_complex_t      *ckappa_d;                                                      // A complex version of kappa
    user_complex_t      tcfx, tcfy, tcfz;                                               // Time correlation function, polarized, ir
    user_complex_t      tcf_iiFii, tcf_iiFjj, tcf_ijFij;                                // Time correlation function, polarized, raman
    user_complex_t      dcy, tcftmp;                                                    // Decay constant and a temporary variable for the tcf
    user_complex_t      *pdtcf, *pdtcf_d;                                               // padded time correlation functions
    user_complex_t      *tcf;                                                           // Time correlation function IR
    user_complex_t      *tcfvv;                                                         // Time correlation function VV raman
    user_complex_t      *tcfvh;                                                         // Time correlation function VH raman
    user_real_t         *Ftcf, *Ftcf_d;                                                 // Fourier transformed time correlation function
    user_real_t         *Ftcfvv, *Ftcfvh;

    // For fft on gpu
    cufftHandle         plan;

    // for timing and errors
    time_t              start=time(NULL), end;
    cudaError_t         Cuerr;
    int                 Merr;
    size_t              freem, total;
    int                 ALLOCATE_2DGPU_ONCE = 0;

    // for file output
    FILE *rtcf;
    FILE *itcf;
    FILE *spec_density;
    FILE *freq_dist;
    FILE *ipr_freq_dist;
    FILE *mipr_freq_dist;
    FILE *spec_lineshape; 
    FILE *vv_lineshape; 
    FILE *vv_rtcf;
    FILE *vv_itcf;
    FILE *vh_lineshape; 
    FILE *vh_rtcf;
    FILE *vh_itcf;
    char *fname;
    fname = (char *) malloc( strlen(outf) + 9 );
    user_real_t factor;                                                                 // conversion factor to give energy and correct intensity from FFT
    user_real_t freq;
    

    // **************************************************** //
    // ***         End  Variable Declaration            *** //


    



    // ***          Begin main routine                  *** //
    // **************************************************** //


    // Open trajectory file and get info about the systeem
    XDRFILE *trj = xdrfile_open( gmxf, "r" ); 
    if ( trj == NULL )
    {
        printf("WARNING: The file %s could not be opened. Is the name correct?\n", gmxf);
        exit(EXIT_FAILURE);
    }

    read_xtc_natoms( (char *)gmxf, &natoms);
    nmol         = natoms / natom_mol;
    nchrom       = nmol * nchrom_mol;
    nchrom2      = (magma_int_t) nchrom*nchrom;
    if ( nchrom < 6000 ) ALLOCATE_2DGPU_ONCE = 1;

    printf(">>> Will read the trajectory from: %s.\n",gmxf);
    printf(">>> Found %d atoms and %d molecules.\n",natoms, nmol);
    printf(">>> Found %d chromophores.\n",nchrom);


    // ***              MEMORY ALLOCATION               *** //
    // **************************************************** //

    // determine the number of blocks to launch on the gpu 
    // each thread takes care of one chromophore for building the electric field and Hamiltonian
    const int numBlocks = (nchrom+blockSize-1)/blockSize;
    
    // Initialize magma math library and queue
    magma_init(); magma_queue_create( 0, &queue ); 

    // CPU arrays
    x       = (rvec*)            malloc( natoms       * sizeof(x[0] ));             if ( x == NULL )      MALLOC_ERR;
    tcf     = (user_complex_t *) calloc( ntcfpoints   , sizeof(user_complex_t));    if ( tcf == NULL )    MALLOC_ERR;
    tcfvv   = (user_complex_t *) calloc( ntcfpoints   , sizeof(user_complex_t));    if ( tcfvv == NULL )  MALLOC_ERR;
    tcfvh   = (user_complex_t *) calloc( ntcfpoints   , sizeof(user_complex_t));    if ( tcfvh == NULL )  MALLOC_ERR;
    Ftcf    = (user_real_t *)    calloc( ntcfpointsR  , sizeof(user_real_t));       if ( Ftcf == NULL )   MALLOC_ERR;
    Ftcfvv  = (user_real_t *)    calloc( ntcfpointsR  , sizeof(user_real_t));       if ( Ftcfvv == NULL ) MALLOC_ERR;
    Ftcfvh  = (user_real_t *)    calloc( ntcfpointsR  , sizeof(user_real_t));       if ( Ftcfvh == NULL ) MALLOC_ERR;


    // GPU arrays
    Cuerr = cudaMalloc( &x_d      , natoms       *sizeof(x[0]));            CHK_ERR;
    Cuerr = cudaMalloc( &eproj_d  , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &Ftcf_d   , ntcfpointsR  *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &mux_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &muy_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &muz_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cmux_d   , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &cmuy_d   , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &cmuz_d   , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &cmux0_d  , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &cmuy0_d  , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &cmuz0_d  , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &tmpmu_d  , nchrom       *sizeof(user_complex_t));  CHK_ERR;
    Cuerr = cudaMalloc( &axx_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &ayy_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &azz_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &axy_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &ayz_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &azx_d    , nchrom       *sizeof(user_real_t));     CHK_ERR;
    Cuerr = cudaMalloc( &caxx_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cayy_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cazz_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &caxy_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cayz_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cazx_d   , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &caxx0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cayy0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cazz0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &caxy0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cayz0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
    Cuerr = cudaMalloc( &cazx0_d  , nchrom       *sizeof(user_complex_t));     CHK_ERR;
 

    // F_d is persistant so alloacate here
    Cuerr = cudaMalloc( &F_d      , nchrom2      *sizeof(user_complex_t)); CHK_ERR;

    // Only allocate temporary non-persistant 2D arrays if the system is small enough
    // Otherwise we have to more actively manage memory to avoid 
    // going over the GPU max memory (4 GB on M1200)
    if ( ALLOCATE_2DGPU_ONCE )
    {
        Cuerr = cudaMalloc( &kappa_d  , nchrom2      *sizeof(user_real_t)); CHK_ERR;
        Cuerr = cudaMalloc( &ckappa_d , nchrom2      *sizeof(user_complex_t)); CHK_ERR;
        Cuerr = cudaMalloc( &ctmpmat_d, nchrom2      *sizeof(user_complex_t)); CHK_ERR;
        Cuerr = cudaMalloc( &prop_d   , nchrom2      *sizeof(user_complex_t)); CHK_ERR;
    }
    kappa   = (user_real_t *)    malloc( nchrom2 * sizeof(user_real_t)); if ( kappa == NULL ) MALLOC_ERR;


    // memory for spectral density calculation
    // CPU arrays
    omega   = (user_real_t *)    malloc( nomega       * sizeof(user_real_t)); if ( omega == NULL ) MALLOC_ERR;
    Sw      = (user_real_t *)    calloc( nomega       , sizeof(user_real_t)); if ( Sw    == NULL ) MALLOC_ERR;
    tmpSw   = (user_real_t *)    malloc( nomega       * sizeof(user_real_t)); if ( tmpSw == NULL ) MALLOC_ERR;
    Pw      = (user_real_t *)    calloc( nomega       , sizeof(user_real_t)); if ( Pw    == NULL ) MALLOC_ERR;
    Rw      = (user_real_t *)    calloc( nomega       , sizeof(user_real_t)); if ( Rw    == NULL ) MALLOC_ERR;
    Rmw     = (user_real_t *)    calloc( nomega       , sizeof(user_real_t)); if ( Rmw    == NULL ) MALLOC_ERR;

    // GPU arrays
    Cuerr = cudaMalloc( &MUX_d   , nchrom       *sizeof(user_real_t)); CHK_ERR;
    Cuerr = cudaMalloc( &MUY_d   , nchrom       *sizeof(user_real_t)); CHK_ERR;
    Cuerr = cudaMalloc( &MUZ_d   , nchrom       *sizeof(user_real_t)); CHK_ERR;
    Cuerr = cudaMalloc( &omega_d , nomega       *sizeof(user_real_t)); CHK_ERR;
    Cuerr = cudaMalloc( &Sw_d    , nomega       *sizeof(user_real_t)); CHK_ERR;
    Cuerr = cudaMalloc( &w_d     , nchrom       *sizeof(user_real_t)); CHK_ERR;

    // initialize omega array
    for (int i = 0; i < nomega; i++) omega[i] = (user_real_t) (omegaStart + omegaStep*i); 
 
    // ***            END MEMORY ALLOCATION             *** //
    // **************************************************** //
    

    // set imodel based on model passed...if 1, reset OM lengths to tip4p lengths
    if ( strcmp( model, "tip4p2005" ) == 0 || strcmp( model, "e3b3" ) == 0 ) imodel = 1;
    else if ( strcmp( model, "tip4p" ) == 0 || strcmp( model, "e3b2" ) == 0 )imodel = 0;
    else{
        printf("WARNING: model: %s is not recognized. Check input file. Aborting...\n", model );
        exit(EXIT_FAILURE);
    }
    // set ispecies based on species passed... 0 H2O, 1 HOD in D2O, 2 HOD in H2O, 3 D2O;
    if ( strcmp( species, "H2O" ) == 0 )        ispecies = 0;
    else if ( strcmp( species, "HOD/D2O" ) == 0 )    ispecies = 1;
    else if ( strcmp( species, "HOD/H2O" ) == 0 )    ispecies = 2;
    else if ( strcmp( species, "D2O" ) == 0 )        ispecies = 3;
    else{
        printf("WARNING: species: %s is not recognized. Check input file. Aborting...\n", species );
        exit(EXIT_FAILURE);
    }

    // index the frames for random access
    read_xtc( trj, natoms, &step, &gmxtime, box, x, &prec );
    float gmxtime2 = gmxtime;
    read_xtc( trj, natoms, &step, &gmxtime, box, x, &prec );
    frame_dt = round((gmxtime-gmxtime2)*prec)/(1.*prec);
    printf(">>> Frame time offset is: %f (ps)\n", frame_dt );
    xdrfile_close(trj);
    printf(">>> Now indexing the xtc file to allow random access.\n");
    read_xtc_n_frames( gmxf, &nframes, &est_nframes, &frame_offset );

    // open xtc file for reading
    trj = xdrfile_open( gmxf, "r" );
    
    printf("\n>>> Now calculating the absorption spectrum\n");
    printf("----------------------------------------------------------\n");

    // **************************************************** //
    // ***          OUTER LOOP OVER SAMPLES             *** //

    while( currentSample < nsamples )
    {
        desired_time = currentSample * sampleEvery + beginTime;
        printf("\n    Now processing sample %d/%d starting at %.2f ps\n",
                currentSample + 1, nsamples, desired_time );
        fflush(stdout);

        // **************************************************** //
        // ***         MAIN LOOP OVER TRAJECTORY            *** //
        while( currentFrame < ntcfpoints )
        {

            // ---------------------------------------------------- //
            // ***          Get Info About The System           *** //


            // read the current frame from the trajectory file and copy to device memory
            // this assumes that the trajectory has no gaps and starts at time zero, but should give a warning if something goes wrong
            desired_time = currentSample * sampleEvery + beginTime + dt * currentFrame;
            int frame = round(desired_time/frame_dt);
            xdrinfo = xdr_seek( trj, frame_offset[ frame ], SEEK_SET ); // set point to beginning of current frame
            //printf("%f\n", desired_time);
            if ( xdrinfo != exdrOK ){
                printf("WARNING:: xdr_seek returned error %d.\n", xdrinfo);
                xdrfile_close(trj); exit(EXIT_FAILURE);
            }
            xdrinfo = read_xtc( trj, natoms, &step, &gmxtime, box, x, &prec ); // read frame from disk
            if ( xdrinfo != exdrOK ){
                printf("Warning:: read_xtc returned error %d.\n", xdrinfo); 
                xdrfile_close(trj); exit(EXIT_FAILURE);
            }
            if ( fabs( desired_time - gmxtime ) > frame_dt*1E-1 ){ // check that we have the frame we want
                printf("\nWARNING:: could not find the desired frame at time %f (ps).\n", desired_time );
                printf("I am instead at gmxtime: %f.\nIs something wrong with the trajectory?", gmxtime );
                exit(EXIT_FAILURE);
            }

            // copy trajectory to gpu memory
            cudaMemcpy( x_d, x, natoms*sizeof(x[0]), cudaMemcpyHostToDevice );

            // allocate space for hamiltonian on the GPU if acively managing GPU memory
            if ( !ALLOCATE_2DGPU_ONCE ) Cuerr = cudaMalloc( &kappa_d  , nchrom2      *sizeof(user_real_t)); CHK_ERR;

            // launch kernel to calculate the electric field projection along OH bonds and build the exciton hamiltonian
            get_eproj_GPU <<<numBlocks,blockSize>>> ( x_d, box[0][0], box[1][1], box[2][2], natoms, natom_mol, nchrom, nchrom_mol, nmol, imodel, eproj_d );
            get_kappa_GPU <<<numBlocks,blockSize>>> ( x_d, box[0][0], box[1][1], box[2][2], natoms, natom_mol, nchrom, nchrom_mol, nmol, eproj_d, kappa_d, 
                                                      mux_d, muy_d, muz_d, axx_d, ayy_d, azz_d, axy_d, ayz_d, azx_d, avef, ispecies, imap);


            // ***          Done getting System Info            *** //
            // ---------------------------------------------------- //




            // ---------------------------------------------------- //
            // ***          Diagonalize the Hamiltonian         *** //

            // Note that kappa only needs to be diagonalized if the exact integration method is requested or the spectral density
            // if the first time, query for optimal workspace dimensions
            if ( SSYEVD_ALLOC_FLAG )
            {
                magma_ssyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) nchrom, NULL, (magma_int_t) nchrom, 
                                  NULL, NULL, (magma_int_t) nchrom, aux_work, -1, aux_iwork, -1, &info );
                lwork   = (magma_int_t) aux_work[0];
                liwork  = aux_iwork[0];

                // allocate work arrays, eigenvalues and other stuff
                w       = (user_real_t *)    malloc( nchrom       * sizeof(user_real_t)); if ( w == NULL ) MALLOC_ERR;

                Merr = magma_imalloc_cpu   ( &iwork, liwork ); CHK_MERR; 
                Merr = magma_smalloc_pinned( &wA , nchrom2 ) ; CHK_MERR;
                Merr = magma_smalloc_pinned( &work , lwork  ); CHK_MERR;
                SSYEVD_ALLOC_FLAG = 0;      // is allocated here, so we won't need to do it again

                // get info about space needed for diagonalization
                cudaMemGetInfo( &freem, &total );
                printf("\n>>> cudaMemGetInfo returned\n"
                       "\tfree:  %g gb\n"
                       "\ttotal: %g gb\n", (float) freem/(1E9), (float) total/(1E9));
                printf(">>> %g gb needed by diagonalization routine.\n", (float) (lwork * (float) sizeof(user_real_t)/(1E9)));
            }

            magma_ssyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) nchrom, kappa_d, (magma_int_t) nchrom,
                              w, wA, (magma_int_t) nchrom, work, lwork, iwork, liwork, &info );
            if ( info != 0 ){ printf("ERROR: magma_dsyevd_gpu returned info %lld.\n", info ); exit(EXIT_FAILURE);}

            // copy eigenvalues to device memory
            cudaMemcpy( w_d    , w    , nchrom*sizeof(user_real_t), cudaMemcpyHostToDevice );

            // ***          Done with the Diagonalization       *** //
            // ---------------------------------------------------- //



            // ---------------------------------------------------- //
            // ***              The Spectral Density            *** //

            if ( currentFrame == 0 )
            {
                // project the transition dipole moments onto the eigenbasis
                // MU_d = kappa_d**T x mu_d 
                magma_sgemv( MagmaTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             1.0, kappa_d, (magma_int_t) nchrom , mux_d, 1, 0.0, MUX_d, 1, queue);
                magma_sgemv( MagmaTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             1.0, kappa_d, (magma_int_t) nchrom, muy_d, 1, 0.0, MUY_d, 1, queue);
                magma_sgemv( MagmaTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             1.0, kappa_d, (magma_int_t) nchrom, muz_d, 1, 0.0, MUZ_d, 1, queue);

                // Initializee the temporary array for spectral density
                for (int i = 0; i < nomega; i++) tmpSw[i] = 0.0;

                // Copy relevant variables to device memory
                cudaMemcpy( omega_d, omega, nomega*sizeof(user_real_t), cudaMemcpyHostToDevice );
                cudaMemcpy( Sw_d   , tmpSw, nomega*sizeof(user_real_t), cudaMemcpyHostToDevice );

                // calculate the spectral density on the GPU and copy back to the CPU
                get_spectral_density <<<numBlocks,blockSize>>> ( w_d, MUX_d, MUY_d, MUZ_d, omega_d, Sw_d, nomega, nchrom, t1, avef );
                cudaMemcpy( tmpSw, Sw_d, nomega*sizeof(user_real_t), cudaMemcpyDeviceToHost );

                // Copy temporary to persistant to get average spectral density over samples
                for (int i = 0; i < nomega; i++ ) Sw[i] += tmpSw[i];
            }

            // ***           Done the Spectral Density          *** //
            // ---------------------------------------------------- //



            // ---------------------------------------------------- //
            // ***              The Frequency Distb.            *** //

            // could make this a function...
            // copy eigenvectors back to host memory
            cudaMemcpy( kappa, kappa_d, nchrom2*sizeof(user_real_t), cudaMemcpyDeviceToHost );

            // loop over eigenstates belonging to the current thread and calculate ipr
            for ( int eign = 0; eign < nchrom; eign ++ ){
                user_real_t c;
                int bin_num;

                // determine ipr
                ipr = 0.; // initialize ipr
                for ( int i = 0; i < nchrom; i ++ ){
                    // calculate ipr
                    c = kappa[eign*nchrom + i];
                    ipr += c*c*c*c;
                }
                ipr = 1./ipr;

                // determine molecular ipr
                user_real_t inner_sum, outer_sum;
                int chrom;
                outer_sum = 0.;
                for ( int i = 0; i < nmol; i ++ ){
                    inner_sum = 0.;  //initialize
                    for ( int j = 0; j < nchrom_mol; j++ ){
                        chrom = i*nchrom_mol + j;
                        c = kappa[eign*nchrom + chrom];
                        inner_sum += c*c;
                    }
                    outer_sum += inner_sum * inner_sum;
                }
                mipr = 1./outer_sum;

                // determine frequency distribution
                wi = w[eign] + avef; // frequency of current mode

                // determine bin number
                bin_num = (int) round((wi - omegaStart)/omegaStep);
                if ( bin_num < 0 || bin_num >= nomega ){
                    printf("WARNING: bin_num is: %d for frequency %g. Check bounds of omegaStart and omegaStop. Aborting.\n", bin_num, wi);
                }

                // divide by omegaStep to make probability density
                Pw[ bin_num] += 1./(omegaStep*1.);
                Rw[ bin_num] += ipr/(omegaStep*1.);
                Rmw[bin_num] += mipr/(omegaStep*1.);
            }

            // ***           Done the Frequency Distb.          *** //
            // ---------------------------------------------------- //


            // ---------------------------------------------------- //
            // ***           Time Correlation Function          *** //


            // allocate space for complex hamiltonian if actively managing memory
            if ( !ALLOCATE_2DGPU_ONCE ) Cuerr = cudaMalloc( &ckappa_d , nchrom2      *sizeof(user_complex_t)); CHK_ERR;

            // cast variables to complex to calculate time correlation function (which is complex)
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( kappa_d, ckappa_d, nchrom2);
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( mux_d  , cmux_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( muy_d  , cmuy_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( muz_d  , cmuz_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( axx_d  , caxx_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( ayy_d  , cayy_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( azz_d  , cazz_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( axy_d  , caxy_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( ayz_d  , cayz_d  , nchrom );
            cast_to_complex_GPU <<<numBlocks,blockSize>>> ( azx_d  , cazx_d  , nchrom );

            
            // free float hamiltonian since we won't need it from here and allocate space for the rest 
            // of the 2D matrix variables that have not yet been allocated if actively managing memory
            if ( !ALLOCATE_2DGPU_ONCE )
            {
                cudaFree( kappa_d );
                Cuerr = cudaMalloc( &ctmpmat_d, nchrom2      *sizeof(user_complex_t)); CHK_ERR;
                Cuerr = cudaMalloc( &prop_d   , nchrom2      *sizeof(user_complex_t)); CHK_ERR;
            }


            // ---------------------------------------------------- //
            // ***           Calculate the F matrix             *** //

            if ( currentFrame == 0 )
            {
                // initialize the F matrix at t=0 to the unit matrix
                makeI <<<numBlocks,blockSize>>> ( F_d, nchrom );

                // set the transition dipole moment at t=0
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( mux_d  , cmux0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( muy_d  , cmuy0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( muz_d  , cmuz0_d  , nchrom );

                // set the polarizability at t=0
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( axx_d  , caxx0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( ayy_d  , cayy0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( azz_d  , cazz0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( axy_d  , caxy0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( ayz_d  , cayz0_d  , nchrom );
                cast_to_complex_GPU <<<numBlocks,blockSize>>> ( azx_d  , cazx0_d  , nchrom );
            }
            else
            {
                // Integrate with exact diagonalization
                // build the propigator
                Pinit <<<numBlocks,blockSize>>> ( prop_d, w_d, nchrom, dt );
                // ctmpmat_d = ckappa_d * prop_d
                magma_cgemm( MagmaNoTrans, MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             (magma_int_t) nchrom, MAGMA_ONE, ckappa_d, (magma_int_t) nchrom, prop_d, 
                             (magma_int_t) nchrom, MAGMA_ZERO, ctmpmat_d, (magma_int_t) nchrom, queue );

                // prop_d = ctmpmat_d * ckappa_d **T 
                magma_cgemm( MagmaNoTrans, MagmaTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             (magma_int_t) nchrom, MAGMA_ONE, ctmpmat_d, (magma_int_t) nchrom, ckappa_d, 
                             (magma_int_t) nchrom, MAGMA_ZERO, prop_d, (magma_int_t) nchrom, queue );

                // ctmpmat_d = prop_d * F
                magma_cgemm( MagmaNoTrans, MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, 
                             (magma_int_t) nchrom, MAGMA_ONE, prop_d, (magma_int_t) nchrom, F_d, 
                             (magma_int_t) nchrom, MAGMA_ZERO, ctmpmat_d, (magma_int_t) nchrom, queue );

                // copy the F matrix back from the temporary variable to F_d
                magma_ccopy( (magma_int_t) nchrom2, ctmpmat_d , 1, F_d, 1, queue );
            }
            // ***           Done updating the F matrix         *** //

            // free 2d matrices if actively managing memory
            if ( !ALLOCATE_2DGPU_ONCE )
            {
                cudaFree( ckappa_d );
                cudaFree( ctmpmat_d );
                cudaFree( prop_d );
            }

            // calculate mFm for x y and z components
            // tcfx = cmux0_d**T * F_d *cmux_d
            // x
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cmux0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcfx = magma_cdotu( (magma_int_t) nchrom, cmux_d, 1, tmpmu_d, 1, queue );

            // y
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cmuy0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcfy = magma_cdotu( (magma_int_t) nchrom, cmuy_d, 1, tmpmu_d, 1, queue );

            // z
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cmuz0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcfz = magma_cdotu( (magma_int_t) nchrom, cmuz_d, 1, tmpmu_d, 1, queue );

            // accumulate the tcf over the samples for the IR spectrum
            tcftmp                = MAGMA_ADD( tcfx  , tcfy );
            tcftmp                = MAGMA_ADD( tcftmp, tcfz );
            tcf[ currentFrame ]   = MAGMA_ADD( tcf[currentFrame], tcftmp );


            // zero variables
            tcf_iiFii = MAGMA_ZERO;
            tcf_ijFij = MAGMA_ZERO;
            tcf_iiFjj = MAGMA_ZERO;

            //              Now The Raman Spectrum             //
            //-------------------------------------------------//
            // tcfxx = caxx0_d**T * F_d * caxx_d
            // **
            // iiFii
            // **

            // xxFxx
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         caxx0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFii = magma_cdotu( (magma_int_t) nchrom, caxx_d, 1, tmpmu_d, 1, queue );

            // yyFyy
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cayy0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFii = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cayy_d, 1, tmpmu_d, 1, queue ), tcf_iiFii );

            // zzFzz
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cazz0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFii = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cazz_d, 1, tmpmu_d, 1, queue ), tcf_iiFii );


            // **
            // ijFij
            // **


            // xyFxy
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         caxy0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_ijFij = magma_cdotu( (magma_int_t) nchrom, caxy_d, 1, tmpmu_d, 1, queue );

            // yzFyz
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cayz0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_ijFij = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cayz_d, 1, tmpmu_d, 1, queue ), tcf_ijFij );

            // zxFzx
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cazx0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_ijFij = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cazx_d, 1, tmpmu_d, 1, queue ), tcf_ijFij );


            // **
            // iiFjj
            // **


            // xxFyy
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         caxx0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = magma_cdotu( (magma_int_t) nchrom, cayy_d, 1, tmpmu_d, 1, queue );

            // xxFzz
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         caxx0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cazz_d, 1, tmpmu_d, 1, queue ), tcf_iiFjj );

            // yyFxx
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cayy0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, caxx_d, 1, tmpmu_d, 1, queue ), tcf_iiFjj);

            // yyFzz
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cayy0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cazz_d, 1, tmpmu_d, 1, queue ), tcf_iiFjj);

            // zzFxx
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cazz0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, caxx_d, 1, tmpmu_d, 1, queue ), tcf_iiFjj);

            // zzFyy
            magma_cgemv( MagmaNoTrans, (magma_int_t) nchrom, (magma_int_t) nchrom, MAGMA_ONE, F_d, (magma_int_t) nchrom,
                         cazz0_d, 1, MAGMA_ZERO, tmpmu_d, 1, queue);
            tcf_iiFjj = MAGMA_ADD( magma_cdotu( (magma_int_t) nchrom, cayy_d, 1, tmpmu_d, 1, queue ), tcf_iiFjj);

            // accumulate the tcf over the samples for the VV raman spectrum
            tcftmp                = MAGMA_ADD( MAGMA_MUL(MAGMA_MAKE(3.,0.), tcf_iiFii), tcf_iiFjj );
            tcftmp                = MAGMA_ADD( tcftmp, MAGMA_MUL(MAGMA_MAKE(4.,0.), tcf_ijFij ));
            tcftmp                = MAGMA_DIV( tcftmp, MAGMA_MAKE(15.,0.) );
            tcfvv[ currentFrame ] = MAGMA_ADD( tcfvv[currentFrame], tcftmp );

            // accumulate the tcf over the samples for the VH raman spectrum
            tcftmp                = MAGMA_ADD( MAGMA_MUL(MAGMA_MAKE(2.,0.), tcf_iiFii), MAGMA_MUL( MAGMA_MAKE(-1.,0.), tcf_iiFjj ));
            tcftmp                = MAGMA_ADD( tcftmp, MAGMA_MUL(MAGMA_MAKE(6.,0.), tcf_ijFij ));
            tcftmp                = MAGMA_DIV( tcftmp, MAGMA_MAKE(30.,0.) );
            tcfvh[ currentFrame ] = MAGMA_ADD( tcfvh[currentFrame], tcftmp );


            // ***        Done with Time Correlation            *** //
            // ---------------------------------------------------- //


            // update progress bar if simulation is big enough, otherwise it really isn't necessary
            if ( nchrom > 400 ) printProgress( currentFrame, ntcfpoints-1 );
            
            // done with current frame, move to next
            currentFrame += 1;
        }

        // done with current sample, move to next, and reset currentFrame to 0
        currentSample +=1;
        currentFrame  = 0;
        
    } // end outer loop

    printf("\n\n----------------------------------------------------------\n");
    printf("Finishing up...\n");

    // close xdr file
    xdrfile_close(trj);


    // ***                  IR Spectrum                 *** //
    // ---------------------------------------------------- //

    // pad the time correlation function with zeros, copy to device memory and perform fft
    // fourier transform the time correlation function on the GPU
    pdtcf = (user_complex_t *) calloc( ntcfpoints+nzeros, sizeof(user_complex_t));
    for ( int i = 0; i < ntcfpoints; i++ )
    {
        // multiply the tcf by the relaxation term
        dcy      = MAGMA_MAKE(exp( -1.0 * i * dt / ( 2.0 * t1 ))/(1.*nsamples), 0.0);
        tcf[i]   = MAGMA_MUL( tcf[i], dcy );
        pdtcf[i] = tcf[i];
    }
    for ( int i = 0; i < nzeros; i++ ) pdtcf[i+ntcfpoints] = MAGMA_ZERO;

    cudaMalloc( &pdtcf_d  , (ntcfpoints+nzeros)*sizeof(user_complex_t));
    cudaMemcpy( pdtcf_d, pdtcf, (ntcfpoints+nzeros)*sizeof(user_complex_t), cudaMemcpyHostToDevice );

    cufftPlan1d  ( &plan, ntcfpoints+nzeros, CUFFT_C2R, 1);
    cufftExecC2R ( plan, pdtcf_d, Ftcf_d );
    cudaMemcpy   ( Ftcf, Ftcf_d, ntcfpointsR*sizeof(user_real_t), cudaMemcpyDeviceToHost );



    // ***                  VV Spectrum                 *** //
    // ---------------------------------------------------- //
    for ( int i = 0; i < ntcfpoints; i++ )
    {
        // multiply the tcf by the relaxation term
        dcy      = MAGMA_MAKE(exp( -1.0 * i * dt / ( 2.0 * t1 ))/(1.*nsamples), 0.0);
        tcfvv[i] = MAGMA_MUL( tcfvv[i], dcy );
        pdtcf[i] = tcfvv[i];
    }
    for ( int i = 0; i < nzeros; i++ ) pdtcf[i+ntcfpoints] = MAGMA_ZERO;
    cudaMemcpy( pdtcf_d, pdtcf, (ntcfpoints+nzeros)*sizeof(user_complex_t), cudaMemcpyHostToDevice );

    cufftExecC2R ( plan, pdtcf_d, Ftcf_d );
    cudaMemcpy   ( Ftcfvv, Ftcf_d, ntcfpointsR*sizeof(user_real_t), cudaMemcpyDeviceToHost );



    // ***                  VH Spectrum                 *** //
    // ---------------------------------------------------- //
    for ( int i = 0; i < ntcfpoints; i++ )
    {
        // multiply the tcf by the relaxation term
        dcy      = MAGMA_MAKE(exp( -1.0 * i * dt / ( 2.0 * t1 ))/(1.*nsamples), 0.0);
        tcfvh[i] = MAGMA_MUL( tcfvh[i], dcy );
        pdtcf[i] = tcfvh[i];
    }
    for ( int i = 0; i < nzeros; i++ ) pdtcf[i+ntcfpoints] = MAGMA_ZERO;
    cudaMemcpy( pdtcf_d, pdtcf, (ntcfpoints+nzeros)*sizeof(user_complex_t), cudaMemcpyHostToDevice );

    cufftExecC2R ( plan, pdtcf_d, Ftcf_d );
    cudaMemcpy   ( Ftcfvh, Ftcf_d, ntcfpointsR*sizeof(user_real_t), cudaMemcpyDeviceToHost );
    cufftDestroy(plan);



    // normalize spectral density by number of samples
    for ( int i = 0; i < nomega; i++) Sw[i]   = Sw[i] / (user_real_t) nsamples;

    // normalize the frequency and ipr weighted frequency distributions
    for ( int i = 0; i < nomega; i ++ ) Pw[i]  /= nchrom*nsamples*ntcfpoints;
    for ( int i = 0; i < nomega; i ++ ) Rw[i]  /= nchrom*nsamples*ntcfpoints;
    for ( int i = 0; i < nomega; i ++ ) Rw[i]  /= Pw[i];
    for ( int i = 0; i < nomega; i ++ ) Rmw[i] /= nchrom*nsamples*ntcfpoints;
    for ( int i = 0; i < nomega; i ++ ) Rmw[i] /= Pw[i];



    // write time correlation function
    rtcf = fopen(strcat(strcpy(fname,outf),"_irrtcf.dat"), "w");
    itcf = fopen(strcat(strcpy(fname,outf),"_iritcf.dat"), "w");
    vv_rtcf = fopen(strcat(strcpy(fname,outf),"_vvrtcf.dat"), "w");
    vv_itcf = fopen(strcat(strcpy(fname,outf),"_vvitcf.dat"), "w");
    vh_rtcf = fopen(strcat(strcpy(fname,outf),"_vhrtcf.dat"), "w");
    vh_itcf = fopen(strcat(strcpy(fname,outf),"_vhitcf.dat"), "w");
    
    
    for ( int i = 0; i < ntcfpoints; i++ )
    {
        fprintf( rtcf, "%g %g \n", i*dt, MAGMA_REAL( tcf[i] ) );
        fprintf( itcf, "%g %g \n", i*dt, MAGMA_IMAG( tcf[i] ) );
        fprintf( vv_rtcf, "%g %g \n", i*dt, MAGMA_REAL( tcfvv[i] ) );
        fprintf( vv_itcf, "%g %g \n", i*dt, MAGMA_IMAG( tcfvv[i] ) );
        fprintf( vh_rtcf, "%g %g \n", i*dt, MAGMA_REAL( tcfvh[i] ) );
        fprintf( vh_itcf, "%g %g \n", i*dt, MAGMA_IMAG( tcfvh[i] ) );
    }
    fclose( rtcf );
    fclose( itcf );
    fclose( vv_rtcf );
    fclose( vv_itcf );
    fclose( vh_rtcf );
    fclose( vh_itcf );



    // write the spectral density
    spec_density = fopen(strcat(strcpy(fname,outf),"_spdn.dat"), "w");
    for ( int i = 0; i < nomega; i++) fprintf(spec_density, "%g %g\n", omega[i], Sw[i]);
    fclose(spec_density);

    // write the frequency distributions
    freq_dist = fopen(strcat(strcpy(fname,outf),"_Pw.dat"), "w");
    for ( int i = 0; i < nomega; i++) fprintf(freq_dist, "%g %g\n", omega[i], Pw[i]);
    fclose(freq_dist);

    ipr_freq_dist = fopen(strcat(strcpy(fname,outf),"_Rw.dat"), "w");
    for ( int i = 0; i < nomega; i++) fprintf(ipr_freq_dist, "%g %g\n", omega[i], Rw[i]);
    fclose(ipr_freq_dist);

    mipr_freq_dist = fopen(strcat(strcpy(fname,outf),"_Rmw.dat"), "w");
    for ( int i = 0; i < nomega; i++) fprintf(mipr_freq_dist, "%g %g\n", omega[i], Rmw[i]);
    fclose(mipr_freq_dist);



    // Write the absorption lineshape
    // Since the C2R transform is inverse by default, the frequencies have to be negated
    // NOTE: to compare with YICUN's code, divide Ftcf by 2
    spec_lineshape = fopen(strcat(strcpy(fname,outf),"_irls.dat"),"w");
    vv_lineshape   = fopen(strcat(strcpy(fname,outf),"_vvls.dat"),"w");
    vh_lineshape   = fopen(strcat(strcpy(fname,outf),"_vhls.dat"),"w");


    factor         = 2*PI*HBAR/(dt*(ntcfpoints+nzeros));                // conversion factor to give energy and correct intensity from FFT
    for ( int i = (ntcfpoints+nzeros)/2; i < ntcfpoints+nzeros; i++ )   // "negative" FFT frequencies
    {
        freq = -1*(i-ntcfpoints-nzeros)*factor + avef;
        if ( freq <= (user_real_t) omegaStop  ) {
            fprintf(spec_lineshape, "%g %g\n", freq, Ftcf[i]/(factor*(ntcfpoints+nzeros)));
            fprintf(vv_lineshape, "%g %g\n", freq, Ftcfvv[i]/(factor*(ntcfpoints+nzeros)));
            fprintf(vh_lineshape, "%g %g\n", freq, Ftcfvh[i]/(factor*(ntcfpoints+nzeros)));
        }
    }
    for ( int i = 0; i < ntcfpoints+nzeros / 2 ; i++)                   // "positive" FFT frequencies
    {
        freq = -1*i*factor + avef;
        if ( freq >= (user_real_t) omegaStart) {
            fprintf(spec_lineshape, "%g %g\n", freq, Ftcf[i]/(factor*(ntcfpoints+nzeros)));
            fprintf(vv_lineshape, "%g %g\n", freq, Ftcfvv[i]/(factor*(ntcfpoints+nzeros)));
            fprintf(vh_lineshape, "%g %g\n", freq, Ftcfvh[i]/(factor*(ntcfpoints+nzeros)));
        }
    }
    fclose(spec_lineshape);
    fclose(vv_lineshape);
    fclose(vh_lineshape);


    // free memory on the CPU and GPU and finalize magma library
    magma_queue_destroy( queue );

    free(x);
    free(Ftcf);
    free(Ftcfvv);
    free(Ftcfvh);
    free(tcf);
    free(tcfvv);
    free(tcfvh);
    free(pdtcf);
    free(Rw);
    free(Pw);
    free(kappa);
    free(Rmw);

    cudaFree(x_d);
    cudaFree(Ftcf_d);
    cudaFree(mux_d); 
    cudaFree(muy_d);
    cudaFree(muz_d);
    cudaFree(eproj_d);
    cudaFree(cmux_d); 
    cudaFree(cmuy_d);
    cudaFree(cmuz_d);
    cudaFree(cmux0_d); 
    cudaFree(cmuy0_d);
    cudaFree(cmuz0_d);
    cudaFree(tmpmu_d);
    cudaFree(axx_d);
    cudaFree(ayy_d);
    cudaFree(azz_d);
    cudaFree(axy_d);
    cudaFree(ayz_d);
    cudaFree(azx_d);
    cudaFree(caxx_d);
    cudaFree(cayy_d);
    cudaFree(cazz_d);
    cudaFree(caxy_d);
    cudaFree(cayz_d);
    cudaFree(cazx_d);
    cudaFree(caxx0_d);
    cudaFree(cayy0_d);
    cudaFree(cazz0_d);
    cudaFree(caxy0_d);
    cudaFree(cayz0_d);
    cudaFree(cazx0_d);
    cudaFree(F_d);

    if ( ALLOCATE_2DGPU_ONCE )
    {
        cudaFree(kappa_d);
        cudaFree(ckappa_d);
        cudaFree(ctmpmat_d);
        cudaFree(prop_d);
    }

    magma_free(pdtcf_d);


    // free memory used for diagonalization
    if ( SSYEVD_ALLOC_FLAG == 0 )
    {
        free(w);
        free(iwork);
        magma_free_pinned( work );
        magma_free_pinned( wA );
    }

    // free memory used in spectral density calculation
    // CPU arrays
    free(omega);
    free(Sw);
    free(tmpSw);

    // GPU arrays
    cudaFree(MUX_d); 
    cudaFree(MUY_d);
    cudaFree(MUZ_d);
    cudaFree(omega_d);
    cudaFree(Sw_d);
    cudaFree(w_d);
 
    // final call to finalize magma math library
    magma_finalize();

    end = time(NULL);
    printf("\n>>> Done with the calculation in %f seconds.\n", difftime(end,start));

    return 0;
}

/**********************************************************
   
   BUILD ELECTRIC FIELD PROJECTION ALONG OH BONDS
                    GPU FUNCTION

 **********************************************************/
__global__
void get_eproj_GPU( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                    int nchrom, int nchrom_mol, int nmol, int model, user_real_t  *eproj )
{
    
    int n, m, i, j, istart, istride;
    int chrom;
    user_real_t mox[XDR_DIM];                     // oxygen position on molecule m
    user_real_t mx[XDR_DIM];                      // atom position on molecule m
    user_real_t nhx[XDR_DIM];                     // hydrogen position on molecule n of the current chromophore
    user_real_t nox[XDR_DIM];                     // oxygen position on molecule n
    user_real_t nohx[XDR_DIM];                    // the unit vector pointing along the OH bond for the current chromophore
    user_real_t mom[XDR_DIM];                     // the OM vector on molecule m
    user_real_t dr[XDR_DIM];                      // the min image vector between two atoms
    user_real_t r;                            // the distance between two atoms 
    const float cutoff = 0.7831;         // the oh cutoff distance
    const float bohr_nm = 18.8973;       // convert from bohr to nanometer
    user_real_t efield[XDR_DIM];                  // the electric field vector

    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // Loop over the chromophores belonging to the current thread
    for ( chrom = istart; chrom < nchrom; chrom += istride )
    {
        // calculate the molecule hosting the current chromophore 
        n = chrom / nchrom_mol;

        // initialize the electric field vector to zero at this chromophore
        efield[0]   =   0.;
        efield[1]   =   0.;
        efield[2]   =   0.;


        // ***          GET INFO ABOUT MOLECULE N HOSTING CHROMOPHORE       *** //
        //                      N IS OUR REFERENCE MOLECULE                     //
        // get the position of the hydrogen associated with the current stretch 
        // NOTE: I'm making some assumptions about the ordering of the positions, 
        // this can be changed if necessary for a more robust program
        // Throughout, I assume that the atoms are grouped into molecules and that
        // every 4th molecule starting at 0 (1, 2, 3) is OW (HW1, HW2, MW)
        if ( chrom % 2 == 0 ){      //HW1
            nhx[0]  = x[ n*natom_mol + 1 ][0];
            nhx[1]  = x[ n*natom_mol + 1 ][1];
            nhx[2]  = x[ n*natom_mol + 1 ][2];
        }
        else if ( chrom % 2 == 1 ){ //HW2
            nhx[0]  = x[ n*natom_mol + 2 ][0];
            nhx[1]  = x[ n*natom_mol + 2 ][1];
            nhx[2]  = x[ n*natom_mol + 2 ][2];
        }

        // The oxygen position
        nox[0]  = x[ n*natom_mol ][0];
        nox[1]  = x[ n*natom_mol ][1];
        nox[2]  = x[ n*natom_mol ][2];

        // The oh unit vector
        nohx[0] = minImage( nhx[0] - nox[0], boxx );
        nohx[1] = minImage( nhx[1] - nox[1], boxy );
        nohx[2] = minImage( nhx[2] - nox[2], boxz );
        r       = mag3(nohx);
        nohx[0] /= r;
        nohx[1] /= r;
        nohx[2] /= r;

        // for testing with YICUN -- can change to ROH later...
        //nohx[0] /= 0.09572;
        //nohx[1] /= 0.09572;
        //nohx[2] /= 0.09572;
 
        // ***          DONE WITH MOLECULE N                                *** //



        // ***          LOOP OVER ALL OTHER MOLECULES                       *** //
        for ( m = 0; m < nmol; m++ ){

            // skip the reference molecule
            if ( m == n ) continue;

            // get oxygen position on current molecule
            mox[0] = x[ m*natom_mol ][0];
            mox[1] = x[ m*natom_mol ][1];
            mox[2] = x[ m*natom_mol ][2];

            // find displacement between oxygen on m and hydrogen on n
            dr[0]  = minImage( mox[0] - nhx[0], boxx );
            dr[1]  = minImage( mox[1] - nhx[1], boxy );
            dr[2]  = minImage( mox[2] - nhx[2], boxz );
            r      = mag3(dr);

            // skip if the distance is greater than the cutoff
            if ( r > cutoff ) continue;

            // loop over all atoms in the current molecule and calculate the electric field 
            // (excluding the oxygen atoms since they have no charge)
            for ( i=1; i < natom_mol; i++ ){

                // position of current atom
                mx[0] = x[ m*natom_mol + i ][0];
                mx[1] = x[ m*natom_mol + i ][1];
                mx[2] = x[ m*natom_mol + i ][2];

                // Move m site to TIP4P distance if model is E3B3 or TIP4P2005 -- this must be done to use the TIP4P map
                if ( i == 3 )
                {
                    if ( model != 0 ) 
                    {
                        // get the OM unit vector
                        mom[0] = minImage( mx[0] - mox[0], boxx );
                        mom[1] = minImage( mx[1] - mox[1], boxy );
                        mom[2] = minImage( mx[2] - mox[2], boxz );
                        r      = mag3(mom);

                        // TIP4P OM distance is 0.015 nm along the OM bond
                        mx[0] = mox[0] + 0.0150*mom[0]/r;
                        mx[1] = mox[1] + 0.0150*mom[1]/r;
                        mx[2] = mox[2] + 0.0150*mom[2]/r;
                    }
                }

                // the minimum image displacement between the reference hydrogen and the current atom
                // NOTE: this converted to bohr so the efield will be in au
                dr[0]  = minImage( nhx[0] - mx[0], boxx )*bohr_nm;
                dr[1]  = minImage( nhx[1] - mx[1], boxy )*bohr_nm;
                dr[2]  = minImage( nhx[2] - mx[2], boxz )*bohr_nm;
                r      = mag3(dr);

                // Add the contribution of the current atom to the electric field
                if ( i < 3  ){              // HW1 and HW2
                    for ( j=0; j < XDR_DIM; j++){
                        efield[j] += 0.52 * dr[j] / (r*r*r);
                    }
                }
                else if ( i == 3 ){         // MW (note the negative sign)
                    for ( j=0; j < XDR_DIM; j++){
                        efield[j] -= 1.04 * dr[j] / (r*r*r);
                    }
                }
            } // end loop over atoms in molecule m

        } // end loop over molecules m

        // project the efield along the OH bond to get the relevant value for the map
        eproj[chrom] = dot3( efield, nohx );

    } // end loop over reference chromophores
}

/**********************************************************
   
   BUILD HAMILTONIAN AND RETURN TRANSITION DIPOLE VECTOR
    FOR EACH CHROMOPHORE ON THE GPU

 **********************************************************/
__global__
void get_kappa_GPU( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, int nchrom, int nchrom_mol, int nmol, 
                    user_real_t *eproj, user_real_t *kappa, user_real_t *mux, user_real_t *muy, user_real_t *muz, user_real_t *axx,
                    user_real_t *ayy, user_real_t *azz, user_real_t *axy, user_real_t *ayz, user_real_t *azx, user_real_t avef, int ispecies,
                    int imap)
{
    
    int n, m, istart, istride;
    int chromn, chromm;
    user_real_t mox[XDR_DIM];                         // oxygen position on molecule m
    user_real_t mhx[XDR_DIM];                         // atom position on molecule m
    user_real_t nhx[XDR_DIM];                         // hydrogen position on molecule n of the current chromophore
    user_real_t nox[XDR_DIM];                         // oxygen position on molecule n
    user_real_t noh[XDR_DIM];
    user_real_t moh[XDR_DIM];
    user_real_t nmu[XDR_DIM];
    user_real_t mmu[XDR_DIM];
    user_real_t mmuprime;
    user_real_t nmuprime;
    user_real_t dr[XDR_DIM];                          // the min image vector between two atoms
    user_real_t r;                                // the distance between two atoms 
    const user_real_t bohr_nm    = 18.8973;       // convert from bohr to nanometer
    const user_real_t cm_hartree = 2.1947463E5;   // convert from cm-1 to hartree
    user_real_t En, Em;                           // the electric field projection
    user_real_t xn, xm, pn, pm;                   // the x and p from the map
    user_real_t wn, wm;                           // the energies

    // define the maps
    user_real_t map_w[3], map_x[2], map_p[2], map_mup[3], map_wi[3];

    // 2013 maps from gruenbaum
    if ( imap == 0 ){
        // H2O and HOD/D2O
        if ( ispecies == 0 || ispecies == 1 ){
            map_w[0] = 3670.2;  map_w[1] = -3541.7; map_w[2] = -152677.0;
            map_x[0] = 0.19285; map_x[1] = -1.7261E-5;
            map_p[0] = 1.6466;  map_p[1] = 5.7692E-4;
        }
        // D2O and HOD/H2O
        if ( ispecies == 2 || ispecies == 3 ){
            map_w[0] = 2767.8;  map_w[1] = -2630.3; map_w[2] = -102601.0;
            map_x[0] = 0.16593; map_x[1] = -2.0632E-5;
            map_p[0] = 2.0475;  map_p[1] = 8.9108E-4;
        }
        map_mup[0] = 0.1646; map_mup[1] = 11.39; map_mup[2] = 63.41;
        map_wi[0] = -1361.0; map_wi[1] = 27165.0; map_wi[2] = -1.887;
    }
    // 2010 map from Li and Skinner
    else if ( imap == 1 )
    {
        // H2O and HOD/D2O
        if ( ispecies == 0 || ispecies == 1 ){
            map_w[0] = 3732.9;  map_w[1] = -3519.8; map_w[2] = -153520.0;
            map_x[0] = 0.19318; map_x[1] = -1.7248E-5;
            map_p[0] = 1.6102;  map_p[1] = 5.8697E-4;
        }
        // D2O and HOD/H2O
        if ( ispecies == 2 || ispecies == 3 ){
            map_w[0] = 2748.2;  map_w[1] = -2572.2; map_w[2] = -102980.0;
            map_x[0] = 0.16598; map_x[1] = -2.0752E-5;
            map_p[0] = 1.9813;  map_p[1] = 9.1419E-4;
        }
        // note the wi have to be converted from hartree to cm from 
        // the values in the table of the paper
        map_mup[0] = 0.1622; map_mup[1] = 10.381; map_mup[2] = 137.6;
        map_wi[0] = -1360.8; map_wi[1] = 27171.0; map_wi[2] = -1.887;
    }

    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // Loop over the chromophores belonging to the current thread and fill in kappa for that row
    for ( chromn = istart; chromn < nchrom; chromn += istride )
    {
        // calculate the molecule hosting the current chromophore 
        // and get the corresponding electric field at the relevant hydrogen
        n   = chromn / nchrom_mol;
        En  = eproj[chromn];

        // get parameters from the map
        wn  = map_w[0] + map_w[1]*En + map_w[2]*En*En;
        xn  = map_x[0] + map_x[1]*wn;
        pn  = map_p[0] + map_p[1]*wn;
        nmuprime = map_mup[0] + map_mup[1]*En + map_mup[2]*En*En;

        // and calculate the location of the transition dipole moment
        // SEE calc_efield_GPU for assumptions about ordering of atoms
        nox[0]  = x[ n*natom_mol ][0];
        nox[1]  = x[ n*natom_mol ][1];
        nox[2]  = x[ n*natom_mol ][2];
        if ( chromn % 2 == 0 )       //HW1
        {
            nhx[0]  = x[ n*natom_mol + 1 ][0];
            nhx[1]  = x[ n*natom_mol + 1 ][1];
            nhx[2]  = x[ n*natom_mol + 1 ][2];
        }
        else if ( chromn % 2 == 1 )  //HW2
        {
            nhx[0]  = x[ n*natom_mol + 2 ][0];
            nhx[1]  = x[ n*natom_mol + 2 ][1];
            nhx[2]  = x[ n*natom_mol + 2 ][2];
        }

        // The OH unit vector
        noh[0] = minImage( nhx[0] - nox[0], boxx );
        noh[1] = minImage( nhx[1] - nox[1], boxy );
        noh[2] = minImage( nhx[2] - nox[2], boxz );
        r      = mag3(noh);
        noh[0] /= r;
        noh[1] /= r;
        noh[2] /= r;

        // The location of the TDM
        nmu[0] = minImage( nox[0] + 0.067 * noh[0], boxx );
        nmu[1] = minImage( nox[1] + 0.067 * noh[1], boxy );
        nmu[2] = minImage( nox[2] + 0.067 * noh[2], boxz );
        
        // and the TDM vector to return
        mux[chromn] = noh[0] * nmuprime * xn;
        muy[chromn] = noh[1] * nmuprime * xn;
        muz[chromn] = noh[2] * nmuprime * xn;

        // and the polarizability
        axx[chromn] = (4.6 * noh[0] * noh[0] + 1.0) * xn;
        ayy[chromn] = (4.6 * noh[1] * noh[1] + 1.0) * xn;
        azz[chromn] = (4.6 * noh[2] * noh[2] + 1.0) * xn;
        axy[chromn] = 4.6 * noh[0] * noh[1] * xn;
        ayz[chromn] = 4.6 * noh[1] * noh[2] * xn;
        azx[chromn] = 4.6 * noh[2] * noh[0] * xn;

        // Loop over all other chromophores
        for ( chromm = 0; chromm < nchrom; chromm ++ )
        {
            // calculate the molecule hosting the current chromophore 
            // and get the corresponding electric field at the relevant hydrogen
            m   = chromm / nchrom_mol;
            Em  = eproj[chromm];

            // also get the relevent x and p from the map
            // get parameters from the map
            wm  = map_w[0] + map_w[1]*Em + map_w[2]*Em*Em;
            xm  = map_x[0] + map_x[1]*wm;
            pm  = map_p[0] + map_p[1]*wm;
            mmuprime = map_mup[0] + map_mup[1]*Em + map_mup[2]*Em*Em;

            // the diagonal energy
            if ( chromn == chromm )
            {
                // Note that this is a flattened 2d array 
                // subtract high frequency energies to get rid of highly oscillatory parts of the F matrix
                kappa[chromn*nchrom + chromm]   = wm - avef;
            }

            // intramolecular coupling
            else if ( m == n )
            {
                // ** -- 
                // if is HOD/H2O or HOD/D2O, no coupling
                if ( ispecies == 1 || ispecies == 2  ){
                    kappa[chromn * nchrom + chromm ] = 0.;
                }
                // ** --
                else{
                    kappa[chromn*nchrom + chromm] = (map_wi[0] + map_wi[1]*(En + Em))*xn*xm + map_wi[2]*pn*pm;
                }
            }

            // intermolecular coupling
            else
            {
                // ** -- 
                // if is HOD/H2O or HOD/D2O, no coupling
                if ( ispecies == 1 || ispecies == 2  ){
                    kappa[chromn * nchrom + chromm ] = 0.;
                }
                // ** --
                
                else{
                    // calculate the distance between dipoles
                    // they are located 0.67 A from the oxygen along the OH bond
                    // tdm position on chromophore n
                    mox[0]  = x[ m*natom_mol ][0];
                    mox[1]  = x[ m*natom_mol ][1];
                    mox[2]  = x[ m*natom_mol ][2];
                    if ( chromm % 2 == 0 )       //HW1
                    {
                        mhx[0]  = x[ m*natom_mol + 1 ][0];
                        mhx[1]  = x[ m*natom_mol + 1 ][1];
                        mhx[2]  = x[ m*natom_mol + 1 ][2];
                    }
                    else if ( chromm % 2 == 1 )  //HW2
                    {
                        mhx[0]  = x[ m*natom_mol + 2 ][0];
                        mhx[1]  = x[ m*natom_mol + 2 ][1];
                        mhx[2]  = x[ m*natom_mol + 2 ][2];
                    }

                    // The OH unit vector
                    moh[0] = minImage( mhx[0] - mox[0], boxx );
                    moh[1] = minImage( mhx[1] - mox[1], boxy );
                    moh[2] = minImage( mhx[2] - mox[2], boxz );
                    r      = mag3(moh);
                    moh[0] /= r;
                    moh[1] /= r;
                    moh[2] /= r;

                    // The location of the TDM and the dipole derivative
                    mmu[0] = minImage( mox[0] + 0.067 * moh[0], boxx );
                    mmu[1] = minImage( mox[1] + 0.067 * moh[1], boxy );
                    mmu[2] = minImage( mox[2] + 0.067 * moh[2], boxz );

                    // the distance between TDM on N and on M and convert to unit vector
                    dr[0] = minImage( nmu[0] - mmu[0], boxx );
                    dr[1] = minImage( nmu[1] - mmu[1], boxy );
                    dr[2] = minImage( nmu[2] - mmu[2], boxz );
                    r     = mag3( dr );
                    dr[0] /= r;
                    dr[1] /= r;
                    dr[2] /= r;
                    r     *= bohr_nm; // convert to bohr

                    // The coupling in the transition dipole approximation in wavenumber
                    // Note the conversion to wavenumber
                    kappa[chromn*nchrom + chromm]   = ( dot3( noh, moh ) - 3.0 * dot3( noh, dr ) * 
                                                        dot3( moh, dr ) ) / ( r*r*r ) * 
                                                        xn*xm*nmuprime*mmuprime*cm_hartree;
                }
            }// end intramolecular coupling
        }// end loop over chromm
    }// end loop over reference
}


/**********************************************************
   
        Calculate the Spectral Density

 **********************************************************/
__global__
void get_spectral_density( user_real_t *w, user_real_t *MUX, user_real_t *MUY, user_real_t *MUZ, user_real_t *omega, user_real_t *Sw, 
                           int nomega, int nchrom, user_real_t t1, user_real_t avef ){

    int istart, istride, i, chromn;
    user_real_t wi, dw, MU2, gamma;
    
    // split up each desired frequency to separate thread on GPU
    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // the linewidth parameter
    gamma = HBAR/(t1 * 2.0);

    // Loop over the chromophores belonging to the current thread and fill in kappa for that row
    for ( i = istart; i < nomega; i += istride )
    {
        // get current frequency
        wi = omega[i];
        
        // Loop over all chromophores calculatint the spectral intensity at the current frequency
        for ( chromn = 0; chromn < nchrom; chromn ++ ){
            // calculate the TDM squared and get the mode energy
            MU2     = MUX[chromn]*MUX[chromn] + MUY[chromn]*MUY[chromn] + MUZ[chromn]*MUZ[chromn];
            dw      = wi - (w[chromn] + avef) ; // also adjust for avef subtracted from kappa

            // add a lorentzian lineshape to the spectral density
            Sw[i] += MU2 * gamma / ( dw*dw + gamma*gamma )/PI;
        }
    }
}


/**********************************************************
   
        HELPER FUNCTIONS FOR GPU CALCULATIONS
            CALLABLE FROM CPU AND GPU

 **********************************************************/



// The minimage image of a scalar
user_real_t minImage( user_real_t dx, user_real_t boxl )
{
    return dx - boxl*round(dx/boxl);
}



// The magnitude of a 3 dimensional vector
user_real_t mag3( user_real_t dx[3] )
{
    return sqrt( dot3( dx, dx ) );
}



// The dot product of a 3 dimensional vector
user_real_t dot3( user_real_t x[3], user_real_t y[3] )
{
    return  x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}



// cast the matrix from float to complex -- this may not be the best way to do this, but it is quick to implement
__global__
void cast_to_complex_GPU ( user_real_t *d_d, user_complex_t *z_d, magma_int_t n )
{
    int istart, istride, i;
    
    // split up each desired frequency to separate thread on GPU
    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // convert from float to complex
    for ( i = istart; i < n; i += istride )
    {
        z_d[i] = MAGMA_MAKE( d_d[i], 0.0 ); 
    }
}

// initialize the propigation matrix
__global__
void Pinit ( user_complex_t *prop_d, user_real_t *w_d, int n, user_real_t dt )
{
    int istart, istride, i, j;
    user_real_t arg;
    
    // each will occour on a separate thread on the gpu
    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    for ( i = istart; i < n; i += istride )
    {
        // zero matrix
        for ( j = 0; j < n; j ++ ) prop_d[ i*n + j] = MAGMA_ZERO;
        // P = exp(iwt/hbar)
        arg   = w_d[i] * dt / HBAR;
        prop_d[ i*n + i ] = MAGMA_MAKE( cos(arg), sin(arg) );
    }
}


// initialize the F matrix on the gpu to the unit matrix
__global__
void makeI ( user_complex_t *mat, int n )
{
    int istart, istride, i, j;
    
    // each will occour on a separate thread on the gpu
    istart  =   blockIdx.x * blockDim.x + threadIdx.x;
    istride =   blockDim.x * gridDim.x;

    // convert from float to complex
    for ( i = istart; i < n; i += istride )
    {
        for ( j = 0; j < n; j++ ) mat[ i*n + j ] = MAGMA_ZERO;
        mat[ i * n + i ] = MAGMA_ONE;
    }
}


// parse input file to setup calculation
void ir_init( char *argv[], char gmxf[], char cptf[], char outf[], char model[], user_real_t *dt, int *ntcfpoints, 
              int *nsamples, float *sampleEvery, user_real_t *t1, user_real_t *avef, user_real_t *omegaStart, user_real_t *omegaStop, 
              int *omegaStep, int *natom_mol, int *nchrom_mol, int *nzeros, user_real_t *beginTime,
              char species[], int *imap )
{
    char                para[MAX_STR_LEN];
    char                value[MAX_STR_LEN];

    FILE *inpf = fopen(argv[1],"r");
    if ( inpf == NULL )
    {
        printf("ERROR: Could not open %s. The first argument should contain  a  vaild\nfile name that points to a file containing the simulation parameters.", argv[1]);
        exit(EXIT_FAILURE);
    }
    else printf(">>> Reading parameters from input file %s\n", argv[1]);

    // Parse input file
    while (fscanf( inpf, "%s%s%*[^\n]", para, value ) != EOF)
    {
        if ( strcmp(para,"xtcf") == 0 ) 
        {
            sscanf( value, "%s", gmxf );
        }
        else if ( strcmp(para,"outf") == 0 )
        {
            sscanf( value, "%s", outf );
        }
        else if ( strcmp(para,"cptf") == 0 ) 
        {
            sscanf( value, "%s", cptf );
        }
        else if ( strcmp(para,"model") == 0 )
        {
            sscanf( value, "%s", model );
        }
        else if ( strcmp(para,"ntcfpoints") == 0 )
        {
            sscanf( value, "%d", (int *) ntcfpoints );
        }
        else if ( strcmp(para,"nsamples") == 0 )
        {
            sscanf( value, "%d", (int *) nsamples);
        }
        else if ( strcmp(para,"sampleEvery") == 0 )
        {
            sscanf( value, "%f", (float *) sampleEvery );
        }
        else if ( strcmp(para,"omegaStep") == 0 )
        {
            sscanf( value, "%d", (int *) omegaStep );
        }
        else if ( strcmp(para,"natom_mol") == 0 )
        {
            sscanf( value, "%d", (int *) natom_mol );
        }
        else if ( strcmp(para,"nchrom_mol") == 0 )
        {
            sscanf( value, "%d", (int *) nchrom_mol );
        }
        else if ( strcmp(para,"nzeros") == 0 )
        {
            sscanf( value, "%d", (int *) nzeros );
        }
        else if ( strcmp(para,"map") == 0 )
        {
            sscanf( value, "%d", (int *) imap );
        }
        else if ( strcmp(para,"species") == 0 )
        {
            sscanf( value, "%s", species );
        }
        else if ( strcmp(para,"dt") == 0 )
        {
            sscanf( value, "%f", dt );
        }
        else if ( strcmp(para,"t1") == 0 )
        {
            sscanf( value, "%f", t1 );
        }
        else if ( strcmp(para,"avef") == 0 )
        {
            sscanf( value, "%f", avef );
        }
        else if ( strcmp(para,"beginTime") == 0 )
        {
            sscanf( value, "%f", beginTime );
        }
        else if ( strcmp(para,"omegaStart") == 0 )
        {
            sscanf( value, "%f", (user_real_t *) omegaStart );
        }
        else if ( strcmp(para,"omegaStop") == 0 )
        {
            sscanf( value, "%f", (user_real_t *) omegaStop );
        }
        else
        {
            printf("WARNING: Parameter %s in input file %s not recognized, ignoring.\n", para, argv[1]);
        }
    }

    fclose(inpf);
    printf(">>> Done reading input file and setting parameters\n");

}



// Progress bar to keep updated on tcf
void printProgress( int currentStep, int totalSteps )
{
    user_real_t percentage = (user_real_t) currentStep / (user_real_t) totalSteps;
    int lpad = (int) (percentage*PWID);
    int rpad = PWID - lpad;
    fprintf(stderr, "\r [%.*s%*s]%3d%%", lpad, PSTR, rpad, "",(int) (percentage*100));
}
