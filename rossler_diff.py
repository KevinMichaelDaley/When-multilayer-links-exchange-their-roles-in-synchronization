#from jax import jacfwd as jacobian
import numpy as np
import sys
#sys.stderr.write(str(xla_bridge.get_backend().platform)+'\n')
sys.stderr.flush()
from scipy.integrate import odeint
def syncdiff(fun, initial_state, m, args, transient_time=0, d0=1e-4, T=2000):
    def system(X,t):
        return fun(t,X,*args)
    yf=odeint((system), initial_state, np.linspace(0,transient_time,transient_time*100))[-1]
    N=len(yf)//m
    yf[m:]=np.array(yf[:m].tolist()*(N-1))
    yf[:m]+=d0*np.random.randn(m)
    Y=odeint((system), yf, np.linspace(0,T,T*10))[-20:]
    """    T=0
    lle=0
    lle_hist=[]
    for i in range(int(tmax//tau)):
        y0=odeint((system), y0, [0,tau], hmax=0.001)[-1]
        d1=np.linalg.norm(y0[m:])
        if d1==0:
            continue
        else:
            T+=tau
            lle+=np.log(d1)
            y0[m:]/=d1
#            sys.stderr.write(str(lle/T)+'\n')
#            sys.stderr.flush()
    return lle/T"""

    sdiff=np.mean(np.abs(Y[:].reshape([-1,1,m])-Y[:].reshape([-1,m,1])))
    return sdiff

if __name__=="__main__":
 def main(pid,nproc):
    def rossler9aux(t,S,ex,ey,a,b,c):
        x,y,z,x2,y2,z2,x3,y3,z3=S
        D1=[-y-z+ex*(x2-x),x+a*y,b+z*x+(x-c)*z]
        D2=[-y2-z2+ex*(x-x2),x2+a*y2+ey*(y3-y2),b+z2*x2+(x2-c)*z2]
        D3=[-y3-z3,x3+a*y3+ey*(y2-y3),b+z3*x3+(x3-c)*z3]
#        D2=[-N1-Z1-2*ex*E1, E1+a*N1+ey*(np.sqrt(3)/2*N2-1/2*N1), z*E1+(x-c)*Z1]
#        D3=[-N2-Z2, E2+a*N2+ey*(np.sqrt(3)/2*N1-3/2*N2), z*E2+(x-c)*Z2]
        return np.array(D1+D2+D3)
    F=lambda t,X,ex,ey: (rossler9aux)(t,X,ex,ey,0.2,0.2,9)
    def O0(e1,e2):
      return syncdiff(F,np.ones(9)*0.1,3,[e1,e2],transient_time=10000,d0=1e-4,T=100)
    for ey in np.linspace(0.0,5,150):
        for ex in np.linspace(0.0,50,150)[pid::nproc]:
            print(ex,ey,O0(ex,ey))
            sys.stdout.flush()
#            for t in range(O[1].shape[1]):
#                for k in range(O[1].shape[0]):
#                   sys.stderr.write(str(O[1][k,t])+' ')
#                sys.stderr.write("\n")
#                sys.stderr.flush()
 main(int(sys.argv[1]), int(sys.argv[2]))



