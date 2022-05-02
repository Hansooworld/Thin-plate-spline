import numpy as np


def get_tpsU(Pt_a, Pt_b):
    D = np.sqrt(np.square(Pt_a[:, None, :2] - Pt_b[None, :, :2]).sum(-1))
    U = D**2 * np.log(D + 1e-6)
    return U

def get_tpscoef(control, target, lmda = 0):
    n = control.shape[0]
    U = get_tpsU(control, control)
    K = U + np.eye(n, dtype=np.float32) * lmda
    P = np.ones((n,3), dtype=np.float32)
    V = np.zeros((n+3,2),dtype=np.float32)
    P[:,1:] = control
    V[:n,:2] = target
    L = np.zeros((n+3,n+3), dtype=np.float32)
    L[:n,:n] = K
    L[:n,-3:] = P
    L[-3:,:n] = P.T
    # print("L Shape", L.shape)
    # print(L)
    # print("V Shape", V.shape)
    # print(V)
    tpscoef = np.linalg.solve(L, V)
    return tpscoef

def tps_trans(source, control, coef):
    n = source.shape[0]
    U = get_tpsU(source, control)
    L = np.hstack([U,np.ones((n,1)),source])
    after = np.dot(L,coef)
    return after