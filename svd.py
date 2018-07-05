import numpy as np
import time

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
def rot(f, g):
    if f == 0:
        c = 0
        s = 1
        r = g
    elif np.abs(f) > np.abs(g):
        t = g / f
        t1 = np.sqrt(1 + t*t)
        c = 1 / t1
        s = t * c
        r = f * t1
    else:
        t = f / g
        t1 = np.sqrt(1 + t*t)
        s = 1 / t1
        c = t * s
        r = g * t1
    return c, s, r

def msweep(X):
    n, _ = np.shape(X)
    U = np.eye(n)
    V = np.eye(n)
    for i in range(n - 1):
        c, s, r = rot(X[i,i], X[i, i+1])
        Q = np.eye(n)
        Q[i:i+2,i:i+2] = np.array([[c, s],[-s, c]])
        X = np.dot(X, Q.T)
        V = np.dot(V, Q.T)
        c, s , r = rot(X[i,i], X[i+1, i])
        Q = np.eye(n)
        Q[i:i+2,i:i+2] = np.array([[c, s],[-s, c]])
        X = np.dot(Q, X)
        U = np.dot(Q, U)
    return U.T, V.T, X

def svd(X):
    n, _ = np.shape(X)
    maxit = 500 * n**2
    thresh = 1e-4

    i_upper = n - 1
    i_lower = 0

    U, V, B  = bidiag_reduction(X)

    for iter in range(20):
        for i in range(i_upper, i_lower, -1):
            i_upper=i
            if np.abs(B[i-1, i]) > thresh:
                break
        for j in range(i_lower, i_upper):
            i_lower=j
            if np.abs(B[j, j+1]) > thresh:
                break
        if i_upper == i_lower and np.abs(e[i_upper]) <= thresh or i_upper < i_lower:
            break
        _U, _V, B = msweep(B)
        U = np.dot(U, _U)
        V = np.dot(_V, V)
    return U, V, B


start = time.time()
a = np.random.rand(100,100)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

U, V, A = svd(a)
print a
print np.dot(np.dot(U, A), V)
end = time.time()
print end - start

start = time.time()
np.linalg.svd(a)
end = time.time()
print end - start
