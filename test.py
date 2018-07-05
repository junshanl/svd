import numpy as np

a = np.random.rand(4,4)

def householder(x):
    n = len(x)
    norm = np.linalg.norm(x)
    e = np.zeros(n)
    e[0] = 1
    v = x - norm * e
    if v.all() != 0:
       v_norm = v / np.linalg.norm(v)
       H = np.eye(n) - 2*np.dot(v_norm[:,np.newaxis], v_norm[np.newaxis, :])
    else:
       H = np.eye(n)
    return H

def fill(X, n):
    m, _ = np.shape(X)
    s = m + n
    A = np.zeros((s,s))
    A[s-m:,s-m:] = X
    for i in range(s-m):
        A[i,i] = 1
    return A

def bidiag_reduction(X):
    m, n = np.shape(X)
    B = X
    U = np.eye(m)
    V = np.eye(n)
    for i in range(m - 1):
        H = householder(B[i:, i])
        H = fill(H, i)
        B = np.dot(H, B)
        U = np.dot(U, H)
        if i +1 < n:
            H = householder(B.T[i+1:, i])
            H = fill(H, i+1)
            B = np.dot(B, H.T)
            V = np.dot(H.T, V)
    return U, V, B

'''
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
U,V,B = bidiag_reduction(a)

print B
print a
print np.dot(np.dot(U, B), V)
'''
