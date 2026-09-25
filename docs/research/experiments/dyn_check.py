exec(open('pga_check.py').read().split('rng = np.random')[0])
import numpy as np
m, Ixx, Iyy, Izz = 2.0, 1.0, 3.0, 5.0
# bivector.net order e01 e02 e03 e12 e31 e23 at idx 5..10 ; I[e01]=m e23, I[e02]=m e31, I[e03]=m e12, I[e12]=Izz e03, I[e31]=Iyy e02, I[e23]=Ixx e01
def Imap(B):
    R=np.zeros(16); R[10]=m*B[5]; R[9]=m*B[6]; R[8]=m*B[7]; R[7]=Izz*B[8]; R[6]=Iyy*B[9]; R[5]=Ixx*B[10]; return R
def Iinv(P):
    R=np.zeros(16); R[5]=P[10]/m; R[6]=P[9]/m; R[7]=P[8]/m; R[8]=P[7]/Izz; R[9]=P[6]/Iyy; R[10]=P[5]/Ixx; return R
def comm(A,B): return 0.5*(GP(A,B)-GP(B,A))
for sgn in (+1,-1):
    def f(M,B):
        return -0.5*GP(M,B), Iinv(sgn*comm(B,Imap(B)))
    M=mv(one=1.0); B=np.zeros(16); B[5:11]=[0.3,-0.2,0.5,0.1,2.0,0.05]
    P0=GP(GP(M,Imap(B)),rev(M)); h=1e-3
    for k in range(3000):
        k1=f(M,B); k2=f(M+h/2*k1[0],B+h/2*k1[1]); k3=f(M+h/2*k2[0],B+h/2*k2[1]); k4=f(M+h*k3[0],B+h*k3[1])
        M=M+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); B=B+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    P1=GP(GP(M,Imap(B)),rev(M))
    E0=None
    print('sign',sgn,' world momentum drift', np.abs(P1-P0).max())
