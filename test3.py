import numpy as np

a = [[0.48,0.82,0.81],[0.96,0.77,0.11],[0.64,0.81,0.96],[0.5,0.5,0.5]]
#a = [[-2.1, 3],[-1, 1.1],[4.3,0.12]]

a = np.array(a)
#aa =  np.dot(a.T,a)
#mu =  np.sum(a, 0)/3
#print (aa - np.dot(mu[:, np.newaxis],mu[np.newaxis,:])) / 2

print np.cov(a.T)

b = a-np.array(np.sum(a, 0)/4)[np.newaxis,:]
print b
print np.dot(b.T, b) 

#print (np.dot(a[:, 0]-0.4, a[:, 0]-0.4))/2
