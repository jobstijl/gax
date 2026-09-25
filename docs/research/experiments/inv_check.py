import numpy as np, itertools
def make(metric):
    n=len(metric); D=1<<n
    def bm(a,b):
        t=a>>1; c=0
        while t: c+=bin(t&b).count('1'); t>>=1
        s=-1.0 if c%2 else 1.0
        for i in range(n):
            if (a&b)>>i&1: s*=metric[i]
        return s,a^b
    tbl=[[bm(i,j) for j in range(D)] for i in range(D)]
    def gp(A,B):
        R=np.zeros(D)
        for i in np.nonzero(A)[0]:
            for j in np.nonzero(B)[0]:
                s,m=tbl[i][j]
                if s: R[m]+=s*A[i]*B[j]
        return R
    g=np.array([bin(i).count('1') for i in range(D)])
    def inv_by(A, signs): return A*signs[g]
    return D,g,gp
def test(metric, trials=5):
    D,g,gp=make(metric); n=len(metric)
    rev=np.array([(-1)**(k*(k-1)//2) for k in range(n+1)]); inv=np.array([(-1)**k for k in range(n+1)]); cc=rev*inv
    R=lambda A:A*rev[g]; H=lambda A:A*inv[g]; C=lambda A:A*cc[g]
    def mneg(A,ks):
        s=np.ones(n+1); 
        for k in ks: s[k]=-1
        return A*s[g]
    rng=np.random.default_rng(0); errs=[]; errsF=[]
    for _ in range(trials):
        x=rng.normal(size=D)
        if n==1: num=H(x)
        elif n==2: num=C(x)
        elif n==3: num=gp(gp(C(x),H(x)),R(x))
        elif n==4: num=gp(C(x),mneg(gp(x,C(x)),[3,4]))
        elif n==5:
            xc=gp(x,C(x)); y=gp(xc,gp(H(x),R(x))) ; num=gp(gp(gp(C(x),H(x)),R(x)),mneg(y,[1,4]))
        den=gp(x,num); errs.append(np.abs(den[1:]).max()/abs(den[0]))
        # Shirokov Faddeev-LeVerrier with N=2^ceil(n/2)
        N=2**((n+1)//2); U=x.copy(); Uk=U.copy()
        for k in range(1,N):
            Ck=N/k*Uk[0]; Uprev=Uk; Uk=gp(U,Uk-Ck*np.eye(D)[0])
        CN=N/N*Uk[0]; det=-CN; adj=(N/(N-1))*Uprev[0]*np.eye(D)[0]-Uprev
        errsF.append(np.abs(gp(x,adj)-det*np.eye(D)[0]).max()/abs(det))
    return max(errs),max(errsF)
for metric in [[1],[0],[1,1],[0,1],[1,1,1],[0,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,-1],[0,1,1,1,-1],[1,1,1,1,-1],[0,0,1,1,1]]:
    print(metric, 'Hitzer residual %.1e  FLV residual %.1e'%test(metric))
