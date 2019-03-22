import numpy as np
import sqaod as sq
from collections import OrderedDict
import pandas as pd

def solver(arr):
	W = np.array(arr)
	output = []
	sol = sq.cpu
	if sq.is_cuda_available() :
	    import sqaod.cuda
	    sol = sqaod.cuda

	ann = sol.dense_graph_annealer()
	ann.seed(1)
	ann.set_qubo(W, sq.minimize)
	ann.set_preferences(n_trotters = W.shape[0])

	h, J, c = ann.get_hamiltonian()
	
	output.append(h); output.append(J); output.append(c)
	
	ann.prepare()
	ann.randomize_spin()

	Ginit = 5.
	Gfin = 0.01
	beta = 1. / 0.02
	tau = 0.99

	G = Ginit
	while Gfin <= G :
	    ann.anneal_one_step(G, beta)
	    G *= tau
	    
	E = ann.get_E()
	q = ann.get_q()
	x = ann.get_x()

	summary = sq.make_summary(ann)

	out_summary = []
	nToShow = min(len(summary.xlist), 4)
	for idx in range(nToShow) :
		out_summary.append(summary.xlist[idx])

	
	output.append(out_summary)
	return output, out_summary




def _main():

	arrays = [[[-0.53256123, -0.02734231,  0.21793854,  0.20448841],
				[-0.02734231,  0.6694462 , -0.10512902, -0.08913602],
				[ 0.21793854, -0.10512902, -0.69904254,  0.24068927],
				[ 0.20448841, -0.08913602,  0.24068927, -0.70646382]], 

				[[-0.53256123, -0.03986413,  0.2203466 ,  0.20860516],
				[-0.03986413,  0.6694462 , -0.07585516, -0.06937777],
				[ 0.2203466 , -0.07585516, -0.69904254,  0.23400084],
				[ 0.20860516, -0.06937777,  0.23400084, -0.70646382]], 

				[[-0.20677455,  0.09348978,  0.22994935,  0.189515],
				[ 0.09348978,  0.99596267,  0.09681199,  0.09499508],
				[ 0.22994935,  0.09681199, -0.37180498,  0.28154435],
				[ 0.189515  ,  0.09499508,  0.28154435, -0.43036846]]]
	ans_vec = []
	bool_vector = []
	for _ in range(100):
		output, out_summary = solver(arrays[0])
		ans_vec.append(output)
		bool_vector.append(out_summary)
		print("Done")

	Save_Vec = np.array(ans_vec)
	np.save('SqaodOutput.npy', Save_Vec)
	df = pd.DataFrame(bool_vector)
	df = df[0].apply(pd.Series).merge(df, left_index=True, right_index=True)
	# df.drop([0], axis=1)
	df.rename(index=str, columns={'0_x':'Q1'}, inplace=True)
	df.rename(index=int, columns={1:'Q2', 2:'Q3', 3:'Q4'}, inplace=True)
	df.drop(columns=['0_y'], inplace=True)
	df.to_csv('Sqaod_Output.csv',index=False)

if __name__ == '__main__':
	_main()
	
