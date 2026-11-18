/* Offline volume-dipole Gauss reference; no fast-math or operational quadrature. */
#include <math.h>
void reference(int np,const double *centers,int nr,const double *pr,const double *wr,
 int ns,const double *ps,const double *ws,const double *u,const double *v,
 double mr,double ms,double *out) {
 for(int p=0;p<np;p++) {
  double *o=out+19*p;
  for(int a=0;a<nr;a++) for(int b=0;b<ns;b++) {
   double r[3],h[3],B[3],F[3],rr=0,du=0,dv=0,uv=0;
   for(int k=0;k<3;k++){r[k]=centers[3*p+k]+pr[3*a+k]-ps[3*b+k];rr+=r[k]*r[k];}
   double rad=sqrt(rr),scale=1e-7*ms*wr[a]*ws[b]/(rr*rad);
   for(int k=0;k<3;k++){h[k]=r[k]/rad;du+=u[k]*h[k];dv+=v[k]*h[k];uv+=u[k]*v[k];}
   for(int k=0;k<3;k++){
    B[k]=scale*(3*h[k]*dv-v[k]);
    F[k]=3*scale*mr/rad*(du*v[k]+dv*u[k]+uv*h[k]-5*du*dv*h[k]);
    o[k]+=B[k];o[3+k]+=F[k];o[9]-=mr*u[k]*B[k];
    for(int j=0;j<3;j++)o[10+3*k+j]+=3*scale/rad*((k==j)*dv+h[k]*v[j]+v[k]*h[j]-5*h[k]*dv*h[j]);
   }
   for(int k=0;k<3;k++){
    int i=(k+1)%3,j=(k+2)%3;
    o[6+k]+=mr*(u[i]*B[j]-u[j]*B[i])+pr[3*a+i]*F[j]-pr[3*a+j]*F[i];
   }
  }
 }
}
