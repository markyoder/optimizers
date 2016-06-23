'''
# some generic ROC tools, and playing a bit with shared memory mpp as well.
# (eventually, this needs to be moved into a general-tools repository). also,
# lets see if we can't @numba.jit  compile some of these bits...
#
# we'll investigate whether to build class structures or take a more proceduarl
# approach -- namely because 1) they don't depend on class-scope variables (input-output),
# 2) we can compile them with @numba.jit, and 3) as per 1,2, they may be more
# portable that way.
'''
###

###
#
import random
import numpy
import scipy
import itertools
import multiprocessing as mpp
#
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
#import functools
import numba
#
def calc_roc_mpp(Z_fc, Z_ev, n_cpu=None, f_denom=None, h_denom=None):
	# mpp handler for roc.
	#
	n_cpu = (n_cpu or mpp.cpu_count())
	f_denom = (f_denom or len(Z_fc))
	h_denom = (h_denom or len(Z_ev))
	#
	# pass all of Z_ev and part of Z_fc to each process. Z_fc can either be parsed by sequential chunks or sampled:
	# X[0:n], X[n:2n], ... or X[0:N:n_cpu], X[1:N:N_cpu],... it shouldn't matter either way, but we have to keep track of their
	# starting indices properly.
	#
	# to be continued...
	
# well... this won't compile either. list inputs maybe? we probalby just need to code these up as extensions... maybe cython...
#@numba.jit
def calc_roc(Z_fc, Z_ev, f_denom=None, h_denom=None, j_fc0=0, j_eq0 do_sort=True):
	# start with a crayon simple 1D ROC.
	#@do_sort: do/don't do the sort. always do the sort unless you're really sure the data are already sorted,
	# like because you did it in an mpp handler or something.
	# j_fc/eq0: starting indices, aka if the job's been parsed up by an mpp handler.
	#
	# maybe make copies? or leave that to the calling function?
	#
	f_denom = float(f_denom or len(Z_fc)+1)
	h_denom = float(h_denom or len(Z_ev)+1)
	#	
	FH=[]
	if do_sort:
		Z_fc.sort()
		Z_ev.sort()
	n_eq=float(len(Z_ev))
	#n_fc=float(len(Z_fc))
	#
	k_eq = 0
	for j,z_fc in enumerate(Z_fc):	
		#
		while k_eq<n_eq and z_fc>=Z_ev[k_eq]:
			#
			#FH += [[j+1,z_fc, k_eq+1, Z_ev[k_eq]]]
			FH += [[(j+1 + j_fc0)/f_denom, (k_eq + j_eq0 +1)/h_denom]]
			#print('ev: ', k_eq, Z_ev[k_eq], z_fc)
			k_eq+=1
		#
	# and one more at the end, just to square off the curve:
	FH += [[(j+1)/f_denom, (k_eq+1)/h_denom]]
	#
	return [[0 for _ in FH[0]]] + FH

def roc_test():
	Z_fc = list(range(10,35))
	Z_eq = list(range(5,25,5))
	Z_eq +=[7,7,7, 20,21,22]
	#
	print('Z_fc: ', Z_fc)
	print('Z_eq: ', Z_eq)
	#
	FH = calc_roc(Z_fc, Z_eq)
	#print('FH: ', FH)
	#
	plt.figure(0)
	plt.clf()
	plt.plot(*zip(*[[j,k] for j,k in FH]), '-', marker='o')
	plt.plot(range(2), range(2), ls='-', color='r', lw=2.)
	#
	return FH

def roc_bench(N=1e6):
	N=int(N)
	R1=random.Random()
	R2=random.Random()
	#
	Z_fc = [R1.random() for _ in range(N)]
	Z_eq = [R2.random() for _ in range(N)]
	#
	# first make sure roc is compiled:
	FH=calc_roc(list(range(5)), list(range(5)))
	#
	t0 = time.time()
	for k in range(100):
		FH = calc_roc(Z_fc, Z_eq)
	#
	print('dt: ', time.time()-t0)

if __name__=='__main__':
	pass
else:
	plt.ion()
#
