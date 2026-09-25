import numpy as np, itertools, math
# PGA3D R(3,0,1), generators e0,e1,e2,e3 with metric 0,1,1,1. Internal: bitmask blades.
metric = [0.0, 1.0, 1.0, 1.0]
def blade_mul(a, b):
    # returns (sign, mask) for canonical-ordered blades a,b (bitmasks, bit i = e_i)
    s = 1.0
    # count swaps
    x = a >> 1
    while x:
        if bin(x & b).count('1') % 2: pass
        x >>= 1
    # standard reordering sign
    t = a >> 1; n = 0
    while t:
        n += bin(t & b).count('1'); t >>= 1
    s = -1.0 if n % 2 else 1.0
    common = a & b
    for i in range(4):
        if common >> i & 1: s *= metric[i]
    return s, a ^ b
def gp(A, B):
    R = np.zeros(16)
    for i in range(16):
        if A[i] == 0: continue
        for j in range(16):
            if B[j] == 0: continue
            s, m = blade_mul(i, j)
            if s: R[m] += s * A[i] * B[j]
    return R
def grade(m): return bin(m).count('1')
def op(A, B):
    R = np.zeros(16)
    for i in range(16):
        for j in range(16):
            if i & j: continue
            s, m = blade_mul(i, j)
            R[m] += s * A[i] * B[j]
    return R
# bivector.net basis names -> (mask, sign) where name = sign * canonical blade
names = ['1','e0','e1','e2','e3','e01','e02','e03','e12','e31','e23','e021','e013','e032','e123','e0123']
def parse(nm):
    if nm == '1': return 0, 1.0
    idx = [int(c) for c in nm[1:]]
    # sign of permutation to sorted
    s = 1.0; l = idx[:]
    for i in range(len(l)):
        for j in range(len(l)-1-i):
            if l[j] > l[j+1]: l[j], l[j+1] = l[j+1], l[j]; s = -s
    m = 0
    for k in idx: m |= 1 << k
    return m, s
tab = [parse(n) for n in names]
def to_int(v):  # bivector.net coeffs -> internal
    R = np.zeros(16)
    for k, (m, s) in enumerate(tab): R[m] += s * v[k]
    return R
def from_int(R):
    return np.array([s * R[m] for (m, s) in tab])
def GP(a, b): return from_int(gp(to_int(a), to_int(b)))
def OP(a, b): return from_int(op(to_int(a), to_int(b)))
def rev(a):
    r = a.copy()
    for k, n in enumerate(names):
        g = 0 if n == '1' else len(n) - 1
        if (g * (g - 1) // 2) % 2: r[k] = -r[k]
    return r
def dual(a): return a[::-1].copy()   # bivector.net dual = coefficient reversal
def vee(a, b): return dual(OP(dual(a), dual(b)))
def mv(**kw):
    v = np.zeros(16)
    for k, x in kw.items(): v[names.index(k if k != 'one' else '1')] = x
    return v
BIV = [5, 6, 7, 8, 9, 10]  # e01 e02 e03 e12 e31 e23
EVEN = [0, 5, 6, 7, 8, 9, 10, 15]
def expm_series(B, n=40):
    R = mv(one=1.0); T = mv(one=1.0)
    for k in range(1, n):
        T = GP(T, B) / k; R = R + T
    return R
# --- De Keninck & Roelfs 2022 listing 2 (basis [1,e01,e02,e03,e12,e31,e23,e0123], biv [e01,e02,e03,e12,e31,e23])
def exp_dkr(b):
    l = b[3]**2 + b[4]**2 + b[5]**2
    if l == 0: return np.array([1, b[0], b[1], b[2], 0, 0, 0, 0.0])
    m = b[0]*b[5] + b[1]*b[4] + b[2]*b[3]; a = math.sqrt(l); c = math.cos(a); s = math.sin(a)/a; t = m/l*(c - s)
    return np.array([c, s*b[0] + t*b[5], s*b[1] + t*b[4], s*b[2] + t*b[3], s*b[3], s*b[4], s*b[5], m*s])
def log_dkr(R):
    if R[0] == 1: return np.array([R[1], R[2], R[3], 0, 0, 0.0])
    a = 1/(1 - R[0]*R[0]); b = math.acos(R[0])*math.sqrt(a); c = a*R[7]*(1 - R[0]*b)
    return np.array([c*R[6] + b*R[1], c*R[5] + b*R[2], c*R[4] + b*R[3], b*R[4], b*R[5], b*R[6]])
def norm_dkr(X):
    A = 1/math.sqrt(X[0]**2 + X[4]**2 + X[5]**2 + X[6]**2)
    B = (X[7]*X[0] - (X[1]*X[6] + X[2]*X[5] + X[3]*X[4]))*A*A*A
    return np.array([A*X[0], A*X[1] + B*X[6], A*X[2] + B*X[5], A*X[3] + B*X[4], A*X[4], A*X[5], A*X[6], A*X[7] - B*X[0]])
def even16(r):
    v = np.zeros(16)
    for k, i in enumerate(EVEN): v[i] = r[k]
    return v
def even8(v): return np.array([v[i] for i in EVEN])
rng = np.random.default_rng(1)
b = rng.normal(size=6)
B16 = np.zeros(16); B16[BIV] = b
E = even8(expm_series(B16))
print('exp closed-form vs series err', np.abs(exp_dkr(b) - E).max())
print('log(exp(b)) err', np.abs(log_dkr(exp_dkr(b*0.3)) - b*0.3).max())
M = exp_dkr(b*0.3); Md = M + rng.normal(size=8)*0.05
N = norm_dkr(Md); N16 = even16(N)
print('normalize: N~N =', np.round(even8(GP(N16, rev(N16))), 12))
S = norm_dkr(np.array([1 + M[0], *M[1:]])); S16 = even16(S)
print('sqrt: S*S - M err', np.abs(even8(GP(S16, S16)) - M).max())
# Mozzi-Chasles: T = 1 + <M>4/<M>2 ; check T commutes with M and T is translator
M16 = even16(M); M2 = np.zeros(16); M2[BIV] = M16[BIV]; M4 = np.zeros(16); M4[15] = M16[15]
# inverse of bivector via Study number: 1/B = ~B (a + b I)^-1
BB = GP(M2, rev(M2)); a_, b_ = BB[0], BB[15]
Binv = GP(rev(M2), mv(one=1/a_, e0123=-b_/a_**2))
print('B*Binv', np.round(GP(M2, Binv), 12)[[0, 15]])
T = mv(one=1.0) + GP(M4, Binv)
print('T is translator? euclid biv parts', np.round(T[[8, 9, 10, 15]], 12), ' [T,M]=', np.abs(GP(T, M16) - GP(M16, T)).max())
# ---- inertia map from point cloud: I[B] = sum m X v (X x B)
def point(x, y, z): return mv(e032=x, e013=y, e021=z, e123=1.0)
def comm(A, B): return 0.5*(GP(A, B) - GP(B, A))
pts = []; mass = 0
for sx, sy, sz in itertools.product([-1, 1], repeat=3):
    pts.append((1.0, point(sx*1.0, sy*2.0, sz*3.0))); mass += 1
def Icloud(Bv): return sum(m * vee(X, comm(X, Bv)) for m, X in pts)
for nm in ['e01', 'e02', 'e03', 'e12', 'e31', 'e23']:
    out = Icloud(mv(**{nm: 1.0}))
    nz = {names[i]: round(out[i], 6) for i in range(16) if abs(out[i]) > 1e-9}
    print('I[', nm, '] =', nz)
print('classical Ixx=sum(y^2+z^2)=', 8*(4+9), 'Iyy=', 8*(1+9), 'Izz=', 8*(1+4), 'mass', mass)
# Kinematics convention check: M(t)=exp(-t B/2) (world). derivative = -1/2 B M
Bw = np.zeros(16); Bw[BIV] = rng.normal(size=6)
h = 1e-6
Mt = lambda t: even16(exp_dkr(-t*Bw[BIV]/2))
dM = (Mt(h) - Mt(-h))/(2*h)
print('dM = -1/2 B M err', np.abs(dM - (-0.5*GP(Bw, Mt(0)))).max())
# Point sandwich: check X' = M X ~M is a point
X = point(0.3, -0.2, 0.7); Xp = GP(GP(M16, X), rev(M16))
print('sandwich point grades ok:', np.abs(Xp[[0,1,2,3,4,5,6,7,8,9,10,15]]).max() < 1e-12, 'e123=', Xp[14])
# Look-Ma-No-Matrices pairing check: exp with e01<-e12 pairing (LMNM swizzle) vs correct
def exp_lmnm(b):  # b in [e01,e02,e03,e12,e31,e23]; LMNM line=[[e23,e31,e12],[e01,e02,e03]]
    L0 = np.array([b[5], b[4], b[3]]); L1 = np.array([b[0], b[1], b[2]])
    l = L0 @ L0; a = math.sqrt(l); m = L0 @ L1; c = math.cos(a); s = math.sin(a)/a; t = m/l*(c - s)
    ideal = s*L1 + t*L0[::-1]    # B[0].zyx
    return np.array([c, *ideal, s*b[3], s*b[4], s*b[5], m*s])
print('LMNM exp (as transcribed) err vs series', np.abs(exp_lmnm(b) - E).max())
