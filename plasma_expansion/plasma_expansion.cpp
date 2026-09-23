// This program calculates the electron distribution at a given input temperature
// and fixed number of escaped electrons.
// This program assumes there is cylindrical Gaussian density
//   exp[-beta*(x^2+y^2) - eta*z^2]
// Edited July 2014 by DAT to make spherical symmetry (beta = eta).
// Sent to Yin Li 05-30-17.

#include <string.h>  // string manipulation
#include <stdio.h>   // file I/O
#include <math.h>    // alternate math functions

// subroutine that initializes the random number generator
void InitRan(long int& i1, long int& i2);

// subroutine that computes the random numbers
long double SimRan1(long int& i1, long int& i2);

// random number generator parameters
const long int ntab  = 32;
const long int ndiv2 = 1 + (2147483563 - 1) / ntab;
long int iv1[ntab];

// --- run parameters: edit these to configure a run ---
// These are read once at the start of main() and set the physical conditions
// being simulated. There are no command-line arguments (inherited from the
// original program); to change a run, edit these constants and rebuild.
const double PARAM_TEMP_K    = 25.0;        // initial electron/ion temperature (K)
const double PARAM_SCALE_M   = 300.E-6;     // characteristic Gaussian cloud size (m); beta0 = eta0 = 1/scale^2
const double PARAM_RHOAVE_M3 = 3.0e8*1.e6;  // average ion/electron number density (m^-3)
const double PARAM_TF_S      = 20.E-6;      // total simulated time (s)
const double PARAM_RYDFRAC   = 2.0;         // number of real atoms each simulated "Rydberg atom" represents
const long int PARAM_INODE   = 1;           // compute-node index; seeds the PRNG so parallel runs differ

int main()
{
    long int i1, i2, inode;

    // these are constants used to set the dimensions of arrays
    // numtim = number of time steps
    // nyrdmax = maximum number of Rydberg "atoms" at a time
    // nstates = maximum number of states in the radiative cascade
    // ndecmax = maximum number of states that any state can radiatively decay into
    const int numtim = 10000, nrydmax = 100000, nstates = 1000, ndecmax = 100;

    // these are integer variables needed in the calculation
    int j, itim, nmod, jj, k, kk, numryd, numatom, numatomp, nstat[nstates],
        lstat[nstates], ndec[nstates], idec[nstates][ndecmax], nstatact,
        n1, l1, n2, l2, latom[nrydmax], nmaxl[ndecmax];

    // these are double real variables used in the calculation
    double pi, scale, rhoave, xnum, ephot, temp, beta0, nstar, melec,
           echrg, numrecomtot, dtim, mion, tf, dum1, kboltz, kinen, potfac,
           totenergy, numrecom, dum2, gamma, beta, expon, dgamma, dexpon, gammap, exponp,
           eps0, wigseitz, debye, coulcoup, energy1, numeryd, eryd,
           dum3, rydprm[6][nrydmax], rydfrac, oneseventh, nstar7, scatf,
           dum, x1, x2, x3, y2, nuryd, estat[nstates], gtot[nstates],
           gdec[nstates][ndecmax], ecut, preftbr, prefl, numatomf, numatomfp, rcut,
           tbrrate, rcut2, rcut3, numatomf2, numatomfp2, numatomf3,
           numatomfp3, eta, eta0, lambda, lambdap, expone, exponep, dlambda,
           dexpone;

    // these characters are used for output
    char fnm2[35], suffix[] = ".dat", tmplt1[] = "plasparRb_", fnm1[35],
         tmplt2[] = "timoutRb_";
         // Original prog from FR 7/22/10 has tmplt1[] = "plaspar0Rb_";
         // changed 12/21/10 to conform with original file names (i.e., FR's on 6/24/10)

    // set the plasma parameters: T(K), scale(m), dens_ave in m^-3, t_final(s)
    // scale changed from 100.E-6 to 700.E-6 on 7/1/14, same date that beta and eta initializations were equalized
    // rhoave = 1.0e9*1.e6 (Nion approx 5e+06 with scale = 700e-06) changed to rhoave = 1.0e8*1.e6 on 7/1/14 (Nion approx 5e+05)
    // eta = beta = 1/scale^2 means that scale = sqrt(2)*sigma and rhoave = (Nion/2*pi*scale^2)^3/2 = (Nion/4*pi*sigma^2)^3/2
    temp = PARAM_TEMP_K; scale = PARAM_SCALE_M; rhoave = PARAM_RHOAVE_M3; tf = PARAM_TF_S;

    // rcut are various radius of cylinders for photon detection
    //    i.e. photons emitted with sqrt(x^2+y^2) < rcut are "detected"
    rcut = scale*3.0; rcut2 = scale*2.0; rcut3 = scale*4.0;

    // the number of atoms each Rydberg "atom" represents
    rydfrac = PARAM_RYDFRAC;


    // START reading in the quantum radiative decay rates

    // read in the states and their n,l
    FILE *inpenrg; inpenrg = fopen("data/energiesH.dat","r");
    if(inpenrg != NULL)
    {
        nstatact = 0;
        while(!feof(inpenrg))
        {
            // read in n, l, E(a.u.)
            fscanf(inpenrg,"%i %i %lf",&nstat[nstatact],&lstat[nstatact],&estat[nstatact]);

            // convert E to J
            estat[nstatact] *= 4.3598e-18;
            nmaxl[lstat[nstatact]] = nstat[nstatact];

            // initialize ndec (number of states that this state decays into) and
            //    gtot (the total radiative rate)
            ndec[nstatact] = 0; gtot[nstatact] = 0.0;

            // initialize the arrays that hold the decay rate into a state (gdec)
            //    and the state that is decayed into (idec)
            for(k = 0 ; k < ndecmax ; k++)
            {
                idec[nstatact][k] = 0; gdec[nstatact][k] = 0.0;
            }
            nstatact ++;
        }
        nstatact --;
//    printf("nstat actual = %i\n",nstatact) ;
    }
    fclose(inpenrg);

    // reorder the states from low to high energy
    for(j = 0 ; j < nstatact-1 ; j++)
    {
        for(k = j+1 ; k < nstatact ; k++)
        {
            if(estat[j] > estat[k])
            {
                jj = lstat[j]; lstat[j] = lstat[k]; lstat[k] = jj;
                jj = nstat[j]; nstat[j] = nstat[k]; nstat[k] = jj;
                pi = estat[j]; estat[j] = estat[k]; estat[k] = pi;
            }
        }
    }

    // read in the rates and associate them with the list of states
    FILE *inprate; inprate = fopen("data/radratesH.dat","r");
    if(inprate != NULL)
    {
        while(!feof(inprate))
        {
            // read in n1,l1 (initial n,l) and n2,l2 (final n,l) and radiative rate
            fscanf(inprate,"%i %i %i %i %lf",&n1,&l1,&n2,&l2,&dum1);
            // find the initial state in the list
            j = 0;
            while((n1 != nstat[j]) || (l1 != lstat[j])) j++;
            // find the final state in the list
            k = 0;
            while((n2 != nstat[k]) || (l2 != lstat[k])) k++;
            // ndec[j] is the # of states that j decays to
            // gdec is the decay rate into each state (gets converted to branching ratio below)
            // gtot is the total decay rate
            // idec[j][ndec] is which is the ndec-th state that j decays to
            gdec[j][ndec[j]] += dum1; gtot[j] += dum1;
            idec[j][ndec[j]] = k; ndec[j] ++;
        }
    }
    fclose(inprate);

    // convert gdec into a cumulated branching ratio
    for(k = 1 ; k < nstatact ; k++)
    {
        gdec[k][0] /= gtot[k];
        for(j = 1 ; j < ndec[k] ; j++) {gdec[k][j] /= gtot[k]; gdec[k][j] += gdec[k][j-1];}
    }

    // ecut is the energy at which the radiative decay starts being calculated
    ecut = estat[((nstatact-1)-4)];  // Modified by DAT 7/26/10; originally: ecut = estat[nstatact-1-4] ;
    //
    // FINISH reading in the radiative decay information



    // set some of the fundamental constants; most are obvious but the
    //    ones that aren't:
    // mion is the mass of the positive ion
    // eryd is the atomic unit of energy (not 1 Rydberg of energy)
    // potfac is the numerator in the calculation of electrostatic potential energy
    pi = 2.*asin(1.); echrg = 1.602e-19; mion = 85.*1.673E-27;
    eryd = 27.21*echrg;
    kboltz = 1.381E-23; eps0 = 8.8542E-12;
    oneseventh = double(1)/double(7); melec = 9.109e-31;
    potfac = echrg*echrg/(4.0*pi*eps0);

    // set initial conditions
    gamma = 0.; lambda = 0.0; expon = 0.0; expone = 0.0;
    // set the initial beta and eta (currently length is 10X more than r) - DAT: this is FR's original code
    // set the initial beta and eta (until 7/1/14 length was 10X more than r) - "beta0 = 1.0/(scale*scale) ; eta0 = 1.0/(100*scale*scale) ;"
    beta0 = 1.0/(scale*scale); eta0 = 1.0/(scale*scale);
    numrecomtot = 0.0; numryd = 0; numrecom = 0.0;
    numatom = 0; numatomp = 0;
    numatomf = 0.0; numatomfp = 0.0;
    numatomf2 = 0.0; numatomfp2 = 0.0;
    numatomf3 = 0.0; numatomfp3 = 0.0;

    // compute some of the parameters of the run
    // xnum is the number of electrons (or ions)
    // dtim is the time step
    xnum = rhoave*(2.*pi/beta0)*sqrt(2.*pi/eta0);
    dtim = tf/numtim;

    // initialize the random number generator
    inode = PARAM_INODE;
    i1 = 5+2836*(inode+100); i2 = 123+12211*(inode+100); InitRan(i1,i2);


    // set the number of times to write output
    nmod = numtim/100;

    // START the naming of the output files
    // set the name of the output file using density, temperature and scale
    dum1 = (log(rhoave)/log(10.0))-6.0+0.01; j = int(dum1);

    jj = j+6; dum1 = rhoave/pow(10.0,jj) + 0.01; jj = int(dum1);

    itim = j+6; dum1 = rhoave-double(jj)*pow(10.0,itim);
    itim  = j+5; dum2 = dum1/pow(10.0,itim) + 0.01; itim = int(dum2);

    dum1 = temp + 0.1; k = int(dum1);

    dum1 = scale/1.e-6 + 0.1; kk = int(dum1);

    sprintf(fnm2,"%s%i.%iE%i_%i_%i%s",tmplt1,jj,itim,j,k,kk,suffix);
    sprintf(fnm1,"%s%i.%iE%i_%i_%i%s",tmplt2,jj,itim,j,k,kk,suffix);
    FILE *outfile; outfile = fopen(fnm2,"w");
    FILE *outtim; outtim = fopen(fnm1,"w");
    printf("name of output file = %s\n",fnm2);
    // FINISH setting name of output file


    // compute the photon energy and the total energy in the plasma
    ephot = temp*kboltz*1.5; totenergy = ephot*(xnum-numrecomtot);
    // convert temp from Kelvin to Joules
    temp *= kboltz;


    // output some of the parameters of the run to the screen
    printf("number of ions = %15.4lf \n",xnum);
    printf("temperature in K = %15.8E \n",temp/kboltz);
    dum1 = 3./(4.*pi*rhoave); wigseitz = pow(dum1,1./3.);
    printf("Wigner-Seitz length %15.8E \n",wigseitz);
    debye=sqrt(temp*eps0/(echrg*echrg*rhoave));
    printf("Debye length %15.8E \n", debye);
    coulcoup = echrg*echrg/(4.*pi*eps0*wigseitz*temp);
    printf("Coulomb coupling coupling parameter %15.8E \n",coulcoup);
    dum = potfac/temp;
    tbrrate = 0.76*rhoave*rhoave*sqrt(temp/9.109e-31)*pow(dum,5);
    printf("TBR Rate = %13.6E\n",tbrrate);
    dum1 = 4.35e-18/temp;
    dum = 7.0*rhoave*pow(dum1,0.17)*0.529e-10*0.529e-10*(3.e8/137.)*(11./3.83)*pow((0.25*dum1),1.33);
    printf("Scat Rate = %13.6E\n",dum);



    // initialize the kinetic energy of the ions to 0
    kinen = 0.;
    // the itim loop is the main time stepping loop
    for(itim = 1 ; itim <= numtim ; itim++)
    {
        // compute beta, the factor in the exponent
        beta = beta0*exp(-expon); eta = eta0*exp(-expone);
        // energy of 1 electron
        energy1 = totenergy/(xnum-numrecomtot);
        // electron temperature
        temp = 2.*(energy1-kinen)/3.;

        // TBR step from Muller and Wolf Eqs 31-33
        // compute the number of TBR during time step
        // nstar is the n_cut off, nstar7 is nstar^7
        nstar = sqrt(13.6*echrg/(2*temp)); nstar7 = pow(nstar,7);
        // dum2 is the factor that comes from integrating the Gaussian density
        //    with the density dependence of the TBR rate
        dum2 = (xnum-numrecomtot)*beta/(sqrt(3.0)*pi);
        dum3 = (xnum-numrecomtot)* eta/(sqrt(3.0)*pi);

        // numrecom is the number of recombined atoms during this time step
        //    plus any fractional electrons from previous steps
        numrecom += nstar7*2.8e-42*dum2*dum2*dum3*dtim*(echrg/temp);

        // convert to integer number of Rydberg "atoms"
        dum2 = numrecom/rydfrac; jj = int(dum2); numrecom -= double(jj)*rydfrac;

        // numrecomtot is the total number of recombined atoms
        numrecomtot += double(jj)*rydfrac;

        // this loop computes the properties of each new Rydberg atom using the
        //    n^6 distribution of Muller and Wolf, the radial distribution of
        //    recombination, and the radial velocity at that r
        for(j = 0 ; j < jj ; j++)
        {
            //   rydprm[0][numryd] is the energy of the atom
            //   rydprm[1][numryd] is the radial position of the atom
            //   rydprm[2][numryd] is the radial velocity of the atom
            //   rydprm[3][numryd] is the z position of the atom
            //   rydprm[4][numryd] is the z velocity of the atom
            //   rydprm[5][numryd] is the time when the atom formed
            // compute the energy of the Rydberg atom randomly formed
            dum2 = SimRan1(i1,i2)*nstar7+5.4e5; dum3 = pow(dum2,oneseventh) - 0.5;
            rydprm[0][numryd] = -2.18E-18/(dum3*dum3);
            // compute the l of the atom formed
            dum = sqrt(SimRan1(i1,i2))*dum3; latom[numryd] = int(dum);
            // shift the total energy of the electron gas + KE of ions by amount from TBR
            totenergy -= rydprm[0][numryd]*rydfrac;
            // compute the radial position where Rydberg formed
            dum2 = sqrt(-log(SimRan1(i1,i2))/(3.0*beta));
            rydprm[1][numryd] = dum2;
            // compute radial velocity of Rydberg atom
            rydprm[2][numryd] = dum2*gamma;
            // compute the z position where Rydberg formed
            dum1 = 2.0*pi*SimRan1(i1,i2); dum2 = sqrt(-log(SimRan1(i1,i2))/(3.0* eta))*sin(dum1);
            rydprm[3][numryd] = dum2;
            // compute z velocity of Rydberg atom
            rydprm[4][numryd] = dum2*lambda;
            // shift the total energy of the electron gas + KE of ions by amount of KE of Rydberg
            totenergy -= 0.5*mion*(rydprm[2][numryd]*rydprm[2][numryd]+rydprm[4][numryd]*rydprm[4][numryd])*rydfrac;
            rydprm[5][numryd] = itim*dtim;
            numryd ++;
        }
        // end TBR

        // e - Rydberg scattering, then step Rydberg position
        if(numryd > 0)
        {
            // this prefactor is common to the electron scattering/ionization rate for
            //   every atom
            preftbr = 11.0*sqrt(temp/melec)*(potfac/temp)*(potfac/temp)*dtim*(xnum-numrecomtot)*
                   (beta/pi)*sqrt( eta/pi);

            // this loops over all of the atoms formed
            for(j = 0 ; j < numryd ; j++)
            {
                // the next 4 lines use Mansbach and Keck's formulas to compute the total
                //    electron-Rydberg collision rate
                dum1 = preftbr*exp(-beta*rydprm[1][j]*rydprm[1][j]- eta*rydprm[3][j]*rydprm[3][j]);
                dum3 = -rydprm[0][j]/temp;
                dum2 = 1./pow(dum3,2.33)+1./(3.83*pow(dum3,1.33));
                dum1 *= dum2;
                // dum1 is now the probability for a scattering during this time interval
                x2 = SimRan1(i1,i2);
                // randomly perform a scattering depending on the probability
                if(x2 < dum1)
                {
                    x3 = SimRan1(i1,i2);
                    // scatf is the relative probability for excitation
                    scatf =1./(1.-rydprm[0][j]/(3.83*temp));
                    // randomly excite or deexcite (the else below) depending on probability
                    if(x3 < scatf)
                    {
                        // if excite the atom, y2 is the final energy after the collision
                        y2 = rydprm[0][j] - temp*log(x3/scatf);
                        // y2 > 0 means ionization and the atom is removed from the Rydberg atom list
                        if(y2 > 0.0)
                        {
                            // decrease the number of Rydberg atoms
                            numrecomtot -= rydfrac;
                            // change the total energy of the electrons plus ions
                            totenergy += (0.5*mion*(rydprm[2][j]*rydprm[2][j]+rydprm[4][j]*rydprm[4][j])+rydprm[0][j])*rydfrac;
                            // shift all of the atoms down 1 in the Rydberg list
                            for(k = j ; k < (numryd-1) ; k++)
                            {
                                for(jj = 0 ; jj < 6 ; jj++) rydprm[jj][k] = rydprm[jj][k+1];
                                latom[k] = latom[k+1];
                            }
                            // decrease the index that counts the number of "atoms"
                            numryd --;
                        }
                        // if the atom is not ionized during excitation
                        else
                        {
                            // change the eneryg of the ions plus electrons by energy given to atom
                            totenergy += (rydprm[0][j] - y2)*rydfrac;
                            // store the new value for the energy
                            rydprm[0][j] = y2;
                            // put in the l change of the atom (l is random)
                            dum = sqrt(-SimRan1(i1,i2)*2.18e-18/y2); latom[j] = int(dum);
                        }
                    }
                    // if the atom is de-excited
                    else
                    {
                        // compute the new energy of the atom
                        y2 = rydprm[0][j]*pow(((1.-scatf)/(1.-x3)),0.2611);
                        // change the energy of the ions plus electrons by energy given to atom
                        totenergy += (rydprm[0][j] - y2)*rydfrac;
                        // store new value for the energy
                        rydprm[0][j] = y2;
                        // put in the l change of the atom (l is random)
                        dum = sqrt(-SimRan1(i1,i2)*2.18e-18/y2); latom[j] = int(dum);
                    }
                }//if(x2 < dum1)
                // change the r-position of the atom using the radial velocity
                rydprm[1][j] += dtim*rydprm[2][j];
                rydprm[3][j] += dtim*rydprm[4][j];
            }//for(j = 0 ; j < numryd ; j++)
        }//if(numryd > 0)
        //FINISH e-Rydberg scattering

        // put in l-changing collision (this is basically a guess, used to mix the l
        //       during the radiative cascade)
        prefl = 5.e2*2.e1*0.53e-10*0.53e-10*dtim*(xnum-numrecomtot)*(beta/pi)*sqrt( eta/pi);
        for(j = 0 ; j < numryd ; j++)
        {
            nuryd = sqrt(-2.18E-18/rydprm[0][j]);
            dum1 = prefl*exp(-beta*rydprm[1][j]*rydprm[1][j] - eta*rydprm[3][j]*rydprm[3][j])*nuryd*nuryd*nuryd*nuryd;
            x1 = SimRan1(i1,i2);
            if(x1 < dum1)
            {
                dum = sqrt(-SimRan1(i1,i2)*2.18e-18/rydprm[0][j]); latom[j] = int(dum);
            }
        }
        //FINISH l-changing collision

        // put in charge transfer (this is now commented out, but could be important
        //          in some cases)
        /*
            prefl = dtim*(beta/pi)*sqrt(beta/pi)*(xnum-numrecomtot)*5.0*potfac*potfac ;
            for(j = 0 ; j < numryd ; j++)
            {
            x1 = SimRan1(i1,i2) ;
            dum1 = prefl*(gamma*rydprm[1][j]-rydprm[2][j])
               *exp(-beta*rydprm[1][j]*rydprm[1][j])/(rydprm[0][j]*rydprm[0][j]) ;
             if(x1 < dum1) rydprm[2][j] = gamma*rydprm[1][j] ;
            }
        */

        // put in radiation decay
        if(numryd > 1)
        {
            for(j = 0 ; j < numryd ; j++)
            {
                if(rydprm[0][j] < ecut)
                {
                    nuryd = sqrt(-2.18E-18/rydprm[0][j]) + 0.5;

                    // use the next lines if using non-hydrogenic levels
                    /*
                        if(latom[j] < 3)
                        {
                        nuryd += 2.2 ;
                        if(latom[j] == 1) nuryd += 1.7 ;
                        if(latom[j] == 2) nuryd += 0.3 ;
                        }
                    */
                    jj = int(nuryd);
                    if(jj == latom[j]) latom[j] = jj-1;
                    if(jj <= nmaxl[latom[j]])
                    {
                        k = 0;
                        while((jj != nstat[k]) || (latom[j] != lstat[k])) k++;
                        x1 = gtot[k]*dtim; x2 = SimRan1(i1,i2);
                        if(x2 < x1)
                        {
                            dum1 = SimRan1(i1,i2); kk = 0;
                            while(gdec[k][kk] < dum1) kk++;
                            n1 = idec[k][kk]; kk = n1;
//       printf("went from %i %i to %i %i\n",nstat[k],lstat[k],nstat[kk],lstat[kk]) ;
//       getchar() ;
                            /*
                                if((nstat[k] == 5) && (lstat[k] == 3) && (nstat[kk] == 3) && (lstat[kk] == 2))
                                {
                                dum1 = rcut/rydprm[1][j] ;
                                if(dum1 > 1.0) numatomf += 1.0 ;
                                else numatomf += 1.0 - sqrt(1.0-dum1*dum1) ;
                                }
                                if((nstat[k] == 6) && (lstat[k] == 0) && (nstat[kk] == 4) && (lstat[kk] == 1))
                                {
                                dum1 = rcut/rydprm[1][j] ;
                                if(dum1 > 1.0) numatomf += 1.0 ;
                                else numatomf += 1.0 - sqrt(1.0-dum1*dum1) ;
                                }
                            */
                            // this block shifts Rydberg info if an atom cascades to below n=5
                            if(nstat[kk] < 5)
                            {
                                fprintf(outtim,"%13.6E %13.6E\n",(dtim*itim),rydprm[5][j]);
                                numatom ++;
                                // put in probability the atom is inside a cylinder
                                dum1 = rcut/rydprm[1][j];
                                if(dum1 > 1.0) numatomf += 1.0;
                                else numatomf += 1.0 - sqrt(1.0-dum1*dum1);
                                dum1 = rcut2/rydprm[1][j];
                                if(dum1 > 1.0) numatomf2 += 1.0;
                                else numatomf2 += 1.0 - sqrt(1.0-dum1*dum1);
                                dum1 = rcut3/rydprm[1][j];
                                if(dum1 > 1.0) numatomf3 += 1.0;
                                else numatomf3 += 1.0 - sqrt(1.0-dum1*dum1);
                                for(k = j ; k < (numryd-1) ; k++)
                                {
                                    for(jj = 0 ; jj < 6 ; jj++) rydprm[jj][k] = rydprm[jj][k+1];
                                    latom[k] = latom[k+1];
                                }
                                numryd --;
                            }
                            else
                            {
                                rydprm[0][j] = estat[kk]; latom[j] = lstat[kk];
                            }
                        }
                    }
                }
            }
        }

        // solve the ODE using 2nd order Runge Kutta algorithm
        beta = beta0*exp(-expon);   // factor in the exponent
        eta =  eta0*exp(-expone);   // factor in the exponent
        kinen = 0.5*mion*gamma*gamma/beta + 0.25*mion*lambda*lambda/eta;
        energy1 = totenergy/(xnum-numrecomtot);      // energy per electron
        dgamma = -gamma*gamma+4.*(energy1-kinen)*beta/(3.*mion);  // d gamma/dt
        dexpon = 2.*gamma;                       // d expon factor/dt
        dlambda = -lambda*lambda+4.*(energy1-kinen)* eta/(3.*mion);  // d lambda/dt
        dexpone = 2.*lambda;                       // d expon factor/dt

        gammap = gamma + dgamma*dtim*0.5; exponp = expon + dexpon*dtim*0.5;  // gamma & expon at dt/2
        lambdap = lambda + dlambda*dtim*0.5; exponep = expone + dexpone*dtim*0.5;

        beta = beta0*exp(-exponp);      // factor in the exponent at dt/2
        eta =  eta0*exp(-exponep);      // factor in the exponent at dt/2
        kinen = 0.5*mion*gammap*gammap/beta + 0.25*mion*lambdap*lambdap/eta;    // KE at dt/2
        dgamma = -gammap*gammap+4.*(energy1-kinen)*beta/(3.*mion);  // d gamma/dt at dt/2
        dexpon = 2.*gammap;                     // d expon factor/dt at dt/2
        dlambda = -lambdap*lambdap+4.*(energy1-kinen)* eta/(3.*mion);  // d lambda/dt
        dexpone = 2.*lambdap;                       // d expon factor/dt
        gamma += dgamma*dtim;                   // gamma at dt
        expon += dexpon*dtim;                   // expon at dt
        lambda += dlambda*dtim;                   // lambda at dt
        expone += dexpone*dtim;                   // expone at dt

        kinen = 0.5*mion*gamma*gamma*exp(expon)/beta0 + 0.25*mion*lambda*lambda*exp(expone)/eta0;  // KE per ion at dt
        temp = 2.*(energy1-kinen)/3.;           // temperature at dt

        // every nmod time steps output info from simulation
        if((itim%nmod) == 0)
        {
            beta=beta0*exp(-expon); eta = eta0*exp(-expone);
            rhoave=(xnum-numrecomtot)*(beta*0.5/pi)*sqrt( eta*0.5/pi);
            dum1 = 3./(4.*pi*rhoave);
            wigseitz = pow(dum1,0.333333);
            debye=sqrt(temp*eps0/(echrg*echrg*rhoave));      // debye length
            coulcoup = echrg*echrg/(4.*pi*eps0*wigseitz*temp);  // coulomb coupling constant
            numeryd = 0.;

            fprintf(outfile,"%13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n",
               (itim-0.5*nmod)*dtim,wigseitz,debye,coulcoup,temp/kboltz,numryd*rydfrac,
               (numatom-numatomp)*rydfrac
               ,1.0/sqrt(beta),1.0/sqrt(eta));
            printf("%10.3E %10.3E %11.4E %11.4E %11.4E %i %i %i %10.3E\n",
               itim*dtim,wigseitz,debye,coulcoup,temp/kboltz,numryd,numatom,numatom-numatomp,
               (numatomf-numatomfp)*rydfrac);
            numatomp = numatom; numatomfp = numatomf; numatomfp2 = numatomf2; numatomfp3 = numatomf3;
        }
    }//for(itim = 1 ; itim <= numtim ; itim++)

}



// function to initiate the random number generator
void InitRan(long int& i1, long int& i2)
{
    int iran;
    for(iran = 0 ; iran < 152 ; iran++)
    {
        i1 = 16807*(i1%127773) - 2836*(i1/127773); if(i1 <= 0) i1+= 2147483647;
        i2 = 40014*(i2%53668) - 12211*(i2/53668);  if(i2 <= 0) i2+= 2147483563;
    }
    for(iran = 0 ; iran < ntab ; iran++)
    {
        i1 = 16807*(i1%127773) - 2836*(i1/127773); if(i1 <= 0) i1+= 2147483647;
        i2 = 40014*(i2%53668) - 12211*(i2/53668);  if(i2 <= 0) i2+= 2147483563;
        iv1[iran] = i1;
    }
}



// random number generator
long double SimRan1(long int& i1, long int& i2)
{
    long int k, j1;
    i1 = 16807*(i1%127773) - 2836*(i1/127773); if(i1 <= 0) i1+= 2147483647;
    i2 = 40014*(i2%53668) - 12211*(i2/53668);  if(i2 <= 0) i2+= 2147483563;
    k = i2/ndiv2; j1 = iv1[k]; iv1[k] = i1;
    return ((j1 - 0.5)*4.656612875245797E-10);
}
