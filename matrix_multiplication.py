import numpy as np
import random

def produce_matrix(vec,m,n):
	l = len(vec)
	if (l != m*n ):
		print("incorrect number of dimensions")
		return []
	output = np.zeros(shape=(m,n))
	row =0
	for i in range(l):
		col = i%n
		output[row][col] = vec[i]		
		if col == n-1:
			row+=1
	return output
			
def multiply_matrices(A,B):
	dim1 = np.shape(A)
	dim2 = np.shape(B) 
	if((dim1[1] != dim2[0])):
		print("Incorrect number of dimensions")
		return []	
	C = np.dot(A,B)
	return C

def power_method_finite(matrix,vector,k):
	n = len(vector)
	vector =vector.reshape((n,1))
	power = vector
	for i in range(k):
		power = np.dot(matrix,power)
		norm = np.linalg.norm(power,ord=2)
		power = power/norm
	return power

def create_markov_setting(Q,d):
	n = len(Q)
	c = np.sum(Q,axis=0)
	for i in range(len(Q)):
		for j in range(len(Q[0])):
			Q[i][j] = Q[i][j]/c[j]
	e = np.ones(shape=(n,n))
	output = d*Q+((1-d)/n)*e
	return output

def get_vector(n):
	return np.array([random.random() for i in range(n)])

if __name__=="__main__":
	x = get_vector(4)
	vec = [0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0]
	matrix = produce_matrix(vec,4,4)
	A = create_markov_setting(matrix,0.85)
	power = power_method_finite(A,x,20)
	import pdb;pdb.set_trace()
	print("power,A,matrix,vec,x")
