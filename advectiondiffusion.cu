# include <stdio.h>      /* printf, scanf, puts, NULL */
# include <cstdlib>      /* srand, rand */
# include <assert.h>     /* assert */
# include <cmath>
# include <algorithm>
# include <iostream>
# include <cassert>
# include <stdint.h>


#include <time.h>
# if not defined(WIN32) && not defined(__USE_POSIX199309)
#   include <sys/time.h>
# endif

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "walltime.c"


void init( int* ndim_tab, int* dim, double* T0, double* x , double* y, double* dx );
double *InitGPUVector(long int N);

void condition_limite(int* ndim_tab, double* T, int nfic);


__global__ void mise_a_jour( int* ndim_tab,   double* T0, double* T1, double* bilan, const double dt );
__global__ void advection( int* ndim_tab,   double* T, double* bilan, double* dx, double* a, int step  );
__global__ void diffusion( int* ndim_tab,   double* T, double* bilan, double* dx, const double mu );
__global__ void condition_limite(int* ndim_tab, double* T, int nfic);


int main( int nargc, char* argv[])
{
  char fileName[255];
  FILE* out;

  int dim[2]; dim[0] = 500; dim[1]=500;
  int nfic     =  2;

  // Comment on gère les fichiers de sorties?
  sprintf(fileName, "Sortie.txt");
  out = fopen(fileName, "w");

  double startTime,elapsedTime;
  double clockZero =0.;
  cudaError_t err = cudaSuccess;

  //
  //
  //Determination la taille des grilles dans les direction X ( Ndim_tab[0]) et Y ( Ndim_tab[1]) avec les cellules fantomes
  //
  //
  int Ndim_tab[2];
  Ndim_tab[0] = dim[0]+2*nfic; 
  Ndim_tab[1] = dim[1]+2*nfic;  

  
  double *x,*y, *T1, *T0,  *bilan;
  double *T1_GPU, *T0_GPU, *bilan_GPU;
  x       = new double[Ndim_tab[0]];
  y       = new double[Ndim_tab[1]];
  bilan   = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  T1      = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  T0      = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  
  double dx[2];
  /* 1- Generation des donnees sur le CPU (host)*/
  init( Ndim_tab, dim, T0, x, y, dx);
  fprintf(out, "dim blocX =  %d, dim blocY =  %d, dx= %f, dy= %f \n",Ndim_tab[0], Ndim_tab[1],  dx[0], dx[1] );

  for (int64_t j = 0; j < Ndim_tab[1] ; ++j ){ 
    for (int64_t i = 0; i < Ndim_tab[0] ; ++i ){ 
      
      int    l = j*Ndim_tab[0]+ i;
      fprintf(out, " Init: %f %f %f   \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Init: \n"); 
  }

  /* 2- Transfert CPU (host) vers GPU (device)*/
  /* 2-a) Allocation memoire sur GPU */
  cudaDeviceSynchronize();
  bilan_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);
  T1_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);
  T0_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);


  /* 2-b) Transfert sur GPU */

  const double dt =0.005;  // pas de temps
  double U[2];
  U[0]  =1.;      // vitesse advection
  U[1]  =1.;
 
  const double mu =0.0005;   // coeff diffusion
  int Nitmax      =2000;
  int Stepmax     = 2;

  //Boucle en temps
  for (int64_t nit = 0; nit < Nitmax ; ++nit )
  { 
    //Boucle Runge-Kutta
    double *Tin;
    double *Tout;
    double *Tbilan;
    for (int64_t step = 0; step < Stepmax ; ++step )
    {
      //mise a jour point courant
      if(step==0) { Tin = T0; Tout= T1; Tbilan= T0;}
      else        { Tin = T0; Tout= T0; Tbilan= T1;}

      //advection
      advection(Ndim_tab, Tbilan, bilan,  dx, U , step);

      diffusion(Ndim_tab, Tbilan, bilan,  dx, mu);
      mise_a_jour(Ndim_tab, Tin, Tout, bilan,  dt);

      //Application Condition limite
      condition_limite(Ndim_tab, Tout, nfic);     

    }  // Nstepmax
  }  // Nitmax

  for (int64_t j = 0; j < Ndim_tab[1] ; ++j ){ 
    for (int64_t i = 0; i < Ndim_tab[0] ; ++i ){ 

      int    l = j*Ndim_tab[0]+ i;
      fprintf(out, " Final %f %f %f  \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Final \n"); 
  }

  fclose(out);

  delete [] T0;  delete [] T1; delete [] bilan; delete [] x;

  return EXIT_SUCCESS;
}



double *InitGPUVector(long int N){
  cudaError_t err = cudaSuccess;
  double *d_x;
  err=cudaMalloc((void **)&d_x, N*sizeof(double));
  if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  return d_x;
}



////
//// Init
////
void init( int* ndim_tab, int* dim, double* T0, double* x , double* y, double* dx )
{
  const double lx = 10.;
  const double ly = 10.;

  dx[0] = lx/dim[0];
  dx[1] = ly/dim[1];

  const double x0 = 0;
  const double y0 = 0;
  const double xinit = 5;
  const double yinit = 5;

  for (int64_t i = 0; i < ndim_tab[0] ; ++i ){
    x[i] = (i-2)*dx[0] + x0;
    for (int64_t j = 0; j < ndim_tab[1] ; ++j ){
      y[j] = (j-2)*dx[1] + y0;

      int l = j*ndim_tab[0]+ i;

      double r = std::sqrt( (x[i]-xinit)*(x[i]-xinit) + (y[j]-yinit)*(y[j]-yinit) );
      T0[l] =300+ 10 * std::exp(-r/0.2);
    }
  }
}




////
//// condition_limite
////
void condition_limite(int* ndim_tab, double* T, int nfic) {
  for (int64_t ific = 0; ific < nfic ; ++ific )
  {  
    //periodicite en Jmax et Jmin
    for (int64_t i = 0; i < ndim_tab[0]  ; ++i )
    {  
      //Jmin
      int l0   = ific*ndim_tab[0] +i;
      int l1   = ndim_tab[0]*(ndim_tab[1]-2*nfic +ific) +i;

      T[l0] = T[l1];

      //Jmax
      l0   = ndim_tab[0]*(ndim_tab[1]-nfic +ific) +i;
      l1   = ndim_tab[0]*(nfic +ific) +i;

      T[l0] = T[l1];
    }
  }

  for (int64_t ific = 0; ific < nfic ; ++ific )
  { 
    //periodicité en Imax et Imin
    for (int64_t j = 0; j < ndim_tab[1]  ; ++j )
    {  
      //Imin
      int l0   = ific +j*ndim_tab[0]; 
      int l1   = l0 + ndim_tab[0] - 2*nfic;

      T[l0] = T[l1];

      //Imax
      l0   = ific + (j+1)*ndim_tab[0] - nfic;
      l1   = l0 - ndim_tab[0] + 2*nfic;

      T[l0] = T[l1];
    }
  }

}




















//On considère les variables suivantes:
//  . On note ndim_tab[0] : la longueur totale de la grille.
//  . On note ndim_tab[1] : la largeur totale de la grille.
//  . dimBlock(blockDim.x, blockDim.y) à définir préalablement de manière à ce que blockDim.x divise ndim_tab[0] et que blockDim.y divise ndim_tab[1].
//  . dimGrid(ndim_tab[0]/blockDim.x, ndim_tab[1]/blockDim.y)

////
//// mise a jour
////
__global__ void mise_a_jour( int* ndim_tab,   double* T0, double* T1, double* bilan, const double dt )
{
    //On récupère les coordonnées de la colonne à traiter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>1 || i<ndim_tab[0]-2) {
      int l;for (int j=2; j<ndim_tab[1]-2; ++j) {  = T0[l] - dt*bilan[l];
      }
    }
}


////
//// advection
////
__global__ void advection( int* ndim_tab,   double* T, double* bilan, double* dx, double* a, int step  ) {

  //On récupère les coordonnées de la colonne à traiter.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i>1 || i<ndim_tab[0]-2) {
    double c1 = 7./6.;
    double c2 = 1./6.;
    // printf("dx %0.9f %0.9f \n", dx, a*dt );

    int l, l1, l2, l3, l4;

    // 1er sous pas schema Heun
    if (step==0) {
      for (int j=2; j<ndim_tab[1]-2; ++j) {
        l = j*ndim_tab[0]+ i; // (i  , j  )
        l1= l+1;              // (i+1, j  )
        l2= l-1;              // (i-1, j  )
        l3= l-2;              // (i-2, j  )
        l4= l+2;              // (i+2, j  )

        double fm   =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
        double fp   =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

        bilan[l] = a[0]*(fp-fm)/(2.*dx[0]); 

        l1= l+ndim_tab[0];     // (i  , j+1)
        l2= l-ndim_tab[0];     // (i  , j-1)
        l3= l-2*ndim_tab[0];   // (i  , j+2)
        l4= l+2*ndim_tab[0];   // (i  , j-2)

        fm   =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
        fp   =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

        bilan[l] += a[1]*(fp-fm)/(2.*dx[1]); 
      }
    }

    // 2eme sous pas schema Heun
    else {
      for (int j=2; j<ndim_tab[1]-2; ++j) {
        l = j*ndim_tab[0]+ i; // (i  , j  )
        l1= l+1;              // (i+1, j  )
        l2= l-1;              // (i-1, j  )
        l3= l-2;              // (i-2, j  )
        l4= l+2;              // (i+2, j  )

        double fm   =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
        double fp   =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

        bilan[l] = 0.5*( bilan[l] + a[0]*(fp-fm)/(2.*dx[0])) ;

        l1= l+ndim_tab[0];     // (i  , j+1)
        l2= l-ndim_tab[0];     // (i  , j-1)
        l3= l-2*ndim_tab[0];   // (i  , j+2)
        l4= l+2*ndim_tab[0];   // (i  , j-2)

        fm   =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
        fp   =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

        bilan[l] += (a[1]*(fp-fm)/(2.*dx[1]))*0.5; 

      }
    }

  }
}

__global__ void diffusion( int* ndim_tab,   double* T, double* bilan, double* dx, const double mu )
{
    //On récupère les coordonnées de la colonne à traiter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>1 || i<ndim_tab[0]-2) {
      for (int j=2; j<ndim_tab[1]-2; ++j) {
        int    l = j*ndim_tab[0]+ i;// (i  , j  )
        int    l1= l+1;              // (i+1, j  )
        int    l2= l-1;              // (i-1, j  )
        int    l3= l+ndim_tab[0];   // (i  , j+1)
        int    l4= l-ndim_tab[0];   // (i  , j-1)
        bilan[l] = bilan[l] - mu*(  (T[l1]+T[l2]-2*T[l])/(dx[0]*dx[0]) +  (T[l3]+T[l4]-2*T[l])/(dx[1]*dx[1]) ) ;
      }

    }
}


////
//// condition_limite 
////
__global__ void condition_limite(int* ndim_tab, double* T, int nfic) {

    //On récupère les coordonnées du point à traiter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
  
    for (int64_t ific = 0; ific < nfic ; ++ific )
    {  
        //periodicite en Jmax et Jmin
        if(i < ndim_tab[0]){  
            //Jmin
            int l0   = ific*ndim_tab[0] +i;
            int l1   = ndim_tab[0]*(ndim_tab[1]-2*nfic +ific) +i;

            T[l0] = T[l1];

            //Jmax
            l0   = ndim_tab[0]*(ndim_tab[1]-nfic +ific) +i;
            l1   = ndim_tab[0]*(nfic +ific) +i;

            T[l0] = T[l1];
        }
    }

    for (int64_t ific = 0; ific < nfic ; ++ific )
    { 
        //periodicité en Imax et Imin
        if(j < ndim_tab[1])
        {  
            //Imin
            int l0   = ific +j*ndim_tab[0]; 
            int l1   = l0 + ndim_tab[0] - 2*nfic;

            T[l0] = T[l1];

            //Imax
            l0   = ific + (j+1)*ndim_tab[0] - nfic;
            l1   = l0 - ndim_tab[0] + 2*nfic;

            T[l0] = T[l1];
        }
    }
}








