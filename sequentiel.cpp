# include <cstdlib>
# include <cmath>
# include <algorithm>
# include <iostream>
# include <cassert>
# include <stdint.h>


#include <time.h>
# if not defined(WIN32) && not defined(__USE_POSIX199309)
#   include <sys/time.h>
# endif





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

////
//// mise a jour
////
void mise_a_jour( int* ndim_tab,   double* T0, double* T1, double* bilan, const double dt, int step )
{
  double cte_rk =0.5;
  if(step==0) { cte_rk = 1;}
  
  for (int64_t j = 2; j < ndim_tab[1]-2 ; ++j ){ 
    for (int64_t i = 2; i < ndim_tab[0]-2 ; ++i ){ 
      int l = j*ndim_tab[0]+ i;
 
      T1[l] = T0[l] - dt*cte_rk*bilan[l]; 
    }
  }
}


////
//// advection
////
void advection( int* ndim_tab,   double* T, double* bilan, double* dxy, double* a, int step  )
{

  double cte_rk =1;
  if(step==0) { cte_rk = 0;}

  double c1 = 7./6.;
  double c2 = 1./6.;

  for (int64_t j = 2; j < ndim_tab[1]-2 ; ++j ) {
    for (int64_t i = 2; i < ndim_tab[0]-2 ; ++i ) { 
      int l = j*ndim_tab[0]+ i;// (i  , j  )
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

////
//// diffusion
////
void diffusion( int* ndim_tab,   double* T, double* bilan, double* dx, const double mu )
{
  for (int64_t j = 2; j < ndim_tab[1]-2 ; ++j ) {
    for (int64_t i = 2; i < ndim_tab[0]-2 ; ++i ) { 
      int l = j*ndim_tab[0]+ i;// (i  , j  )
      int l1= l+1;              // (i+1, j  )
      int l2= l-1;              // (i-1, j  )
      int l3= l+ndim_tab[0];   // (i  , j+1)
      int l4= l-ndim_tab[0];   // (i  , j-1)
      bilan[l] = bilan[l] - mu*(  (T[l1]+T[l2]-2*T[l])/(dx[0]*dx[0]) +  (T[l3]+T[l4]-2*T[l])/(dx[1]*dx[1]) ) ;
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


int main( int nargc, char* argv[])
{
  char fileName[255];
  FILE* out;

  int dim[2]; dim[0] = 500; dim[1]=500;
  int nfic     =  2;

  int rank=0;
  sprintf(fileName, "Sortie%05d.txt", rank);
  out = fopen(fileName, "w");
 

  //
  //
  //Determination la taille des grilles dans les direction X ( Ndim_tab[0]) et Y ( Ndim_tab[1]) avec les cellules fantomes
  //
  //
  int Ndim_tab[2];
  Ndim_tab[0] = dim[0]+2*nfic; 
  Ndim_tab[1] = dim[1]+2*nfic;  


  double *x,*y, *T1, *T0,  *bilan;
  x       = new double[Ndim_tab[0]];
  y       = new double[Ndim_tab[1]];
  bilan   = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  T1      = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  T0      = new double[Ndim_tab[0]*Ndim_tab[1]]; 
  
  double xy0[2];
  xy0[0]=0;
  xy0[1]=0;
  double lx =10;
  double ly =10;
  double dxy[2];
  dxy[0] = lx/ dim[0];
  dxy[1] = ly/ dim[1];

  init( Ndim_tab, T0, x, y, dxy, xy0);
  fprintf(out, "dim blocX =  %d, dim blocY =  %d, dx= %f, dy= %f \n",Ndim_tab[0], Ndim_tab[1],  dxy[0], dxy[1] );

  for (int64_t i = 0; i < Ndim_tab[0] ; ++i ){ 
    for (int64_t j = 0; j < Ndim_tab[1] ; ++j ){ 
      int l = j*Ndim_tab[0]+ i;
      fprintf(out, " Init: %f %f %f   \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Init: \n"); 
  }



  const double dt =0.004;  // pas de temps
  double U[2];
  U[0]  =1.;      // vitesse advection suivant axe X
  U[1]  =1.;      // vitesse advection suivant axe Y

 
  const double mu =0.0005;   // coeff diffusion
  //int Nitmax      =1250;   // temps final = 5s
  int Nitmax      =1875;     // temps final = 7.5s
  //int Nitmax = 1; // test
  int Stepmax     = 2;       //schema  RK2

  //Boucle en temps
  for (int64_t nit = 0; nit < Nitmax ; ++nit )
  { 
    //Boucle Runge-Kutta
    double *Tin;
    double *Tout;
    double *Tbilan;
    for (int64_t step = 0; step < Stepmax ; ++step )
    { 

      if(step==0) { Tin = T0; Tout= T1; Tbilan=T0; }
      else        { Tin = T0; Tout= T0; Tbilan=T1;}

      //advection
      advection(Ndim_tab, Tbilan, bilan,  dxy, U , step);
      diffusion(Ndim_tab, Tbilan, bilan,  dxy, mu);

      mise_a_jour(Ndim_tab, Tin, Tout, bilan,  dt, step);

      //Application Condition limite
      condition_limite(Ndim_tab, Tout, nfic);

    }  // Nstepmax
  }  // Nitmax

  for (int64_t i = nfic; i < Ndim_tab[0]-nfic ; ++i ){ 
    for (int64_t j = nfic; j < Ndim_tab[1]-nfic ; ++j ){ 
      
      int    l = j*Ndim_tab[0]+ i;
      fprintf(out, " Final %f %f %f  \n", x[i],y[j], T0[l]); 
    }
    fprintf(out, " Final \n"); 
  }

  fclose(out);

  delete [] T0;  delete [] T1; delete [] bilan; delete [] x;

  return EXIT_SUCCESS;
}
