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

// initialisation à T=0
void init( int* ndim_tab, double* T0, double* x , double* y, double* dxy, double* xy0 );

// initialisation des pointeurs GPU
double *InitGPUVector(long int N);
int *InitGPUVector_int(long int N);

// Fonctions globales
__global__ void mise_a_jour( int* ndim_tab,   double* T0, double* T1, double* bilan, const double dt, int step);
__global__ void advection( int* ndim_tab,   double* T, double* bilan, double* dxy, double* a, int step  );
__global__ void diffusion( int* ndim_tab,   double* T, double* bilan, double* dx, const double mu );
__global__ void condition_limite(int* ndim_tab, double* T, int nfic);



int main( int nargc, char* argv[])
{
  char fileName[255];
  FILE* out;

  int dim[2]; dim[0] = 500; dim[1]=500;
  int nfic     =  2;

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

  double *T1_GPU, *T0_GPU, *bilan_GPU,*U_GPU,*dxy_GPU;
  int *Ndim_tab_GPU;
  double *x,*y, *T0;
  x       = new double[Ndim_tab[0]];
  y       = new double[Ndim_tab[1]];
  T0      = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  
  double xy0[2];
  xy0[0]=0;
  xy0[1]=0;
  double lx =10;
  double ly =10;
  double dxy[2];
  dxy[0] = lx/ dim[0];
  dxy[1] = ly/ dim[1];


  const double dt =0.004;  // pas de temps
  double U[2];
  U[0]  =1.;      // vitesse advection suivant axe X
  U[1]  =1.;      // vitesse advection suivant axe Y

 
  const double mu =0.0005;   // coeff diffusion
  //int Nitmax      =1250;   // temps final = 5s
  int Nitmax      =1875;     // temps final = 7.5s
  int Stepmax     = 2;       //schema  RK2

  /* 1- Generation des donnees sur le CPU (host)*/
  startTime=walltime(&clockZero);
  init( Ndim_tab, T0, x, y, dxy, xy0);
  elapsedTime=walltime(&startTime);
  printf("Time to generate datas at T=0 : %6.4f(ms)\n", elapsedTime*1000);

  fprintf(out, "dim blocX =  %d, dim blocY =  %d, dx= %f, dy= %f \n",Ndim_tab[0], Ndim_tab[1],  dxy[0], dxy[1] );

  for (int64_t i = 0; i < Ndim_tab[0] ; ++i ){ 
    for (int64_t j = 0; j < Ndim_tab[1] ; ++j ){ 
      int l = j*Ndim_tab[0]+ i;
      fprintf(out, " Init: %f %f %f   \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Init: \n"); 
  }
  elapsedTime=walltime(&startTime);
  printf("Time to write initial condition in Sortie.txt : %6.4f(ms)\n", elapsedTime*1000);

  /* 2- Transfert CPU (host) vers GPU (device)*/
  /* 2-a) Allocation memoire sur GPU */
  cudaDeviceSynchronize();
  bilan_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);
  T1_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);
  T0_GPU = InitGPUVector(Ndim_tab[0]*Ndim_tab[1]);
  
  U_GPU = InitGPUVector(2);
  dxy_GPU = InitGPUVector(2);
  Ndim_tab_GPU = InitGPUVector_int(2);
  

  /* 2-b) Transfert sur GPU */
  size_t size = Ndim_tab[0]*Ndim_tab[1] * sizeof(double);

  err = cudaMemcpy(U_GPU, U, 2*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector U from host to device T0_GPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(dxy_GPU, dxy, 2*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector dxy from host to device T0_GPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(Ndim_tab_GPU, Ndim_tab, 2*sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector Ndim_tab from host to device T0_GPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  err = cudaMemcpy(T0_GPU, T0, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector T0 from host to device T0_GPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  cudaDeviceSynchronize();
  elapsedTime=walltime(&startTime);
  printf("Time for transfert CPU->GPU : %6.4f(ms)\n", elapsedTime*1000);

  /* 3- Calcul sur GPU*/
  int threadsPerBlock = 32; //doit être un multiple de 32
  int blocksPerGrid =(Ndim_tab[0] + threadsPerBlock - 1) / threadsPerBlock;

  //Boucle en temps
  for (int64_t nit = 0; nit < Nitmax ; ++nit )
  { 
    //Boucle Runge-Kutta
    double *Tin;
    double *Tout;
    double *Tbilan;

    for (int64_t step = 0; step < Stepmax ; ++step )
    { 

      if(step==0) { Tin = T0_GPU; Tout= T1_GPU; Tbilan=T0_GPU; }
      else        { Tin = T0_GPU; Tout= T0_GPU; Tbilan=T1_GPU;}

      //advection
      advection<<<blocksPerGrid, threadsPerBlock>>>(Ndim_tab_GPU, Tbilan, bilan_GPU,  dxy_GPU, U_GPU , step);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch advection kernel à la boucle %d, step %d (error code %s)!\n",nit, step, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      diffusion<<<blocksPerGrid, threadsPerBlock>>>(Ndim_tab_GPU, Tbilan, bilan_GPU,  dxy_GPU, mu);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch diffusion kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      mise_a_jour<<<blocksPerGrid, threadsPerBlock>>>(Ndim_tab_GPU, Tin, Tout, bilan_GPU,  dt, step);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mise_a_jour kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      //Application Condition limite
      condition_limite<<<blocksPerGrid, threadsPerBlock>>>(Ndim_tab_GPU, Tout, nfic);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch condition_limite kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    }  // Nstepmax
  }  // Nitmax

  cudaDeviceSynchronize();
  elapsedTime=walltime(&startTime);
  printf("Time for GPU calculus : %6.4f(ms)\n", elapsedTime*1000);

  /* 4- Transfert GPU (device) vers CPU (host)*/
  err = cudaMemcpy(T0, T0_GPU, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  elapsedTime=walltime(&startTime);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector T0_GPU from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  printf("Time for transfert GPU->CPU : %6.4f(ms)\n", elapsedTime*1000);

  /* 5- Sortie  */
  for (int64_t i = nfic; i < Ndim_tab[0]-nfic ; ++i ){ 
    for (int64_t j = nfic; j < Ndim_tab[1]-nfic ; ++j ){ 
      
      int    l = j*Ndim_tab[0]+ i;
      fprintf(out, " Final %f %f %f  \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Final \n"); 
  }

  fclose(out);

  /* 6- Nettoyage */
  delete [] T0; delete [] x; delete [] y;
  cudaFree(bilan_GPU);cudaFree(T1_GPU);cudaFree(T0_GPU);
  cudaFree(U_GPU); cudaFree(dxy_GPU); cudaFree(Ndim_tab_GPU);
  cudaDeviceReset();

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

int *InitGPUVector_int(long int N){
  cudaError_t err = cudaSuccess;
  int *d_x;
  err=cudaMalloc((void **)&d_x, N*sizeof(int));
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
void init( int* ndim_tab, double* T0, double* x , double* y, double* dxy, double* xy0 )
{
  const double x0 = xy0[0] +dxy[0]*0.5;
  const double y0 = xy0[1] +dxy[1]*0.5;
  const double xinit = 5;
  const double yinit = 5;

  for (int64_t i = 0; i < ndim_tab[0] ; ++i ){
    x[i] = (i-2)*dxy[0] + x0;
    for (int64_t j = 0; j < ndim_tab[1] ; ++j ){
      y[j] = (j-2)*dxy[1] + y0;
      
      int l = j*ndim_tab[0]+ i;

      double r = std::sqrt( (x[i]-xinit)*(x[i]-xinit) + (y[j]-yinit)*(y[j]-yinit) );
      T0[l] =300+ 10 * std::exp(-r/0.2);
    }
  }
}






//On considère les variables suivantes:
//  . On note ndim_tab[0] : la longueur totale de la grille.
//  . On note ndim_tab[1] : la largeur totale de la grille.
//  . dimBlock(blockDim.x, blockDim.y) à définir préalablement de manière à ce que blockDim.x divise ndim_tab[0]
//  . dimGrid(ndim_tab[0]/blockDim.x, ndim_tab[1])

////
//// mise a jour
////
__global__ void mise_a_jour( int* ndim_tab,   double* T0, double* T1, double* bilan, const double dt, int step )
{
    //On récupère les coordonnées de la colonne à traiter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>1 && i<ndim_tab[0]-2) {
      double cte_rk =0.5;
      if(step==0) { cte_rk = 1;}

      int l;
      for (int j=2; j<ndim_tab[1]-2; ++j) {
        l = j*ndim_tab[0]+ i;
        T1[l] = T0[l] - dt*cte_rk*bilan[l]; 
      }
    }
}


////
//// advection
////
__global__ void advection( int* ndim_tab,   double* T, double* bilan, double* dxy, double* a, int step  ) {

  //On récupère les coordonnées de la colonne à traiter.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i>1 && i<ndim_tab[0]-2) {
    double cte_rk =1;
    if(step==0) { cte_rk = 0;}

    double c1 = 7./6.;
    double c2 = 1./6.;
    // printf("dx %0.9f %0.9f \n", dx, a*dt );

    for (int j=2; j<ndim_tab[1]-2; ++j) {
      int    l = j*ndim_tab[0]+ i;// (i  , j  )
      int    l1= l+1;              // (i+1, j  )
      int    l2= l-1;              // (i-1, j  )
      int    l3= l-2;              // (i-2, j  )
      int    l4= l+2;              // (i+2, j  )

      double fm   =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
      double fp   =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

      bilan[l] = bilan[l]*cte_rk + a[0]*(fp-fm)/(2.*dxy[0]); 

      l1= l+ndim_tab[0];     // (i  , j+1)
      l2= l-ndim_tab[0];     // (i  , j-1)
      l3= l-2*ndim_tab[0];   // (i  , j-2)
      l4= l+2*ndim_tab[0];   // (i  , j+2)

      fm =(T[l ]+T[l2])*c1 - (T[l1]+T[l3])*c2;
      fp =(T[l1]+T[l ])*c1 - (T[l4]+T[l2])*c2;

      bilan[l] += a[1]*(fp-fm)/(2.*dxy[1]); 
    }

  }
}

__global__ void diffusion( int* ndim_tab,   double* T, double* bilan, double* dx, const double mu )
{
    //On récupère les coordonnées de la colonne à traiter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>1 && i<ndim_tab[0]-2) {
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

  //On récupère les coordonnées de la colonne à traiter.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < ndim_tab[0]){
    //periodicite en Jmax et Jmin
    for (int64_t ific = 0; ific < nfic ; ++ific ) {
      //Jmin
      int l0   = ific*ndim_tab[0] +i;
      int l1   = ndim_tab[0]*(ndim_tab[1]-2*nfic +ific) +i;

      T[l0] = T[l1];

      //Jmax
      l0   = ndim_tab[0]*(ndim_tab[1]-nfic +ific) +i;
      l1   = ndim_tab[0]*(nfic +ific) +i;

      T[l0] = T[l1];
    }

    //periodicité en Imin
    if (i<nfic ) { 
      for (int64_t j = 0; j < ndim_tab[1]  ; ++j ) {  
        //Imin
        int l0   = i +j*ndim_tab[0]; 
        int l1   = l0 + ndim_tab[0] - 2*nfic;
 
        T[l0] = T[l1];
 
      }
    }

    //periodicité en Imax
    if (i>=ndim_tab[0]-nfic) {
      for (int64_t j = 0; j < ndim_tab[1]  ; ++j ) {
        //Imax
        int l0   = i + j*ndim_tab[0];
        int l1   = l0 - ndim_tab[0] + 2*nfic;

        T[l0] = T[l1];
      }
    }

  }

}






