exec(open('pga_check.py').read().split('rng = np.random')[0])
rng=np.random.default_rng(3); b=rng.normal(size=6)
B=np.zeros(16); B[BIV]=b
BB=GP(B,B); dot=BB[0]; wedge=BB[15]
print('B.B', dot, ' -|E|^2', -(b[3]**2+b[4]**2+b[5]**2), ' (B^B)_e0123', wedge, ' 2m', 2*(b[0]*b[5]+b[1]*b[4]+b[2]*b[3]))
# b2 = (B^B)/(2B) = (B^B) B^{-1} / 2
BBv=np.zeros(16); BBv[0]=dot; BBv[15]=wedge
Binv=GP(B, mv(one=1/dot, e0123=-wedge/dot**2))
b2=0.5*GP(mv(e0123=wedge),Binv); b1=B-b2
l=b[3]**2+b[4]**2+b[5]**2; m=b[0]*b[5]+b[1]*b[4]+b[2]*b[3]
print('b2 coeffs [e01,e02,e03]', np.round(b2[[5,6,7]],12), ' formula (m/l)*[b23,b31,b12]', np.round(m/l*np.array([b[5],b[4],b[3]]),12))
print('b1^b1', np.round(GP(b1,b1)[15],12), 'b1 b2 commute', np.abs(GP(b1,b2)-GP(b2,b1)).max(), 'b1^2', GP(b1,b1)[0], 'b2^2', GP(b2,b2)[0])
