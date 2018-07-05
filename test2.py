import numpy as np

a = np.random.randn(3,3)
b = np.random.randn(3,3)
c = np.random.randn(3,3)

e = np.eye(3)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
print np.dot(np.dot(a, b),c)
print np.dot(a,np.dot(b ,c))

print np.dot(a, b)
print np.dot(np.dot(e , b),a)
