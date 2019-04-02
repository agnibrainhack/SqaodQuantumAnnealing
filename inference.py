import numpy as np
import sqaod as sq
from collections import OrderedDict
import pandas as pd

def solver(arr):
	W = arr
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

	data = pd.read_csv('Data/wiscon_mi_mRMR_qubo.csv')
	data = data.drop(data.columns[0], axis=1)
	arrays = data.values
	# arrays = np.array(arrays)
	ans_vec = []
	bool_vector = []
	
	output, out_summary = solver(arrays)
	ans_vec.append(output)
	bool_vector.append(out_summary)
	print("Done")

	Save_Vec = np.array(ans_vec)
	np.save('Answer2/wiscon_mi_mRMR.npy', Save_Vec)
	print(bool_vector)
	df = pd.DataFrame(bool_vector)
	df = df[0].apply(pd.Series).merge(df, left_index=True, right_index=True)
	# df.drop([0], axis=1)
	df.rename(index=str, columns={'0_x':'Q1'}, inplace=True)
	df.rename(index=int, columns={1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5', 5:'Q6', 6:'Q7'}, inplace=True)
	df.drop(columns=['0_y'], inplace=True)
	df.to_csv('Answer2/wiscon_mi_mRMR.csv',index=False)

if __name__ == '__main__':
	_main()
	
