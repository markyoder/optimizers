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
import multiprocessing as mpp
import numba
#
def calc_roc_mpp(Z_fc, Z_ev, n_cpu=None, f_denom=None, h_denom=None):
	# mpp handler for roc.
	# TODO: ok, so this looks like it's working, but a more quantitative test with larget data sets is in order. first, do a point-by-point on the spp roc; then bench mpp vs spp
	# on a large array... also, add a "stepify" function, so spp and mpp curves can be explicitly compared... or write a script to parse out redundant elements (if x[0][0]==x[1][0]
	# or x[0][1]==x[1][1], remove the later one). we get these elements because one sub-process will find the "hit" at a different z_fc than another process.
	#
	n_cpu = (n_cpu or mpp.cpu_count())
	f_denom = float(f_denom or len(Z_fc))
	h_denom = float(h_denom or len(Z_ev))
	#
	# pass all of Z_ev and part of Z_fc to each process. Z_fc can either be parsed by sequential chunks or sampled:
	# X[0:n], X[n:2n], ... or X[0:N:n_cpu], X[1:N:N_cpu],... it shouldn't matter either way, but we have to keep track of their
	# starting indices properly.
	#
	P = mpp.Pool(n_cpu)
	p_len = int(numpy.ceil(len(Z_fc)/n_cpu))
	#
	Z_fc.sort()
	Z_ev.sort()
	#
	workers = [P.apply_async(calc_roc, args=(Z_fc[j*p_len:(j+1)*p_len], Z_ev), kwds={'f_denom':f_denom, 'h_denom':h_denom, 'j_fc0':j*p_len, 'j_eq0':0, 'do_sort':False}) for j in range(n_cpu)]
	#	
	P.close()
	P.join()
	#
	roc_segments = [worker.get() for worker in workers]
	#roc_segments = [print('work...') for worker in workers]
	#
	roc = [rw for roc_segment in roc_segments for rw in roc_segment]
	roc.sort(key = lambda rw: rw[0])
	
	return roc
	
# well... this won't compile either. list inputs maybe? we probalby just need to code these up as extensions... maybe cython...
#@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.int64, numba.int64, numba.boolean)], '(n),(n)->(n)')
def calc_roc(Z_fc, Z_ev, f_denom=None, h_denom=None, j_fc0=0, j_eq0=0, do_sort=True):
	# start with a crayon simple 1D ROC.
	#@do_sort: do/don't do the sort. always do the sort unless you're really sure the data are already sorted,
	# like because you did it in an mpp handler or something.
	# j_fc/eq0: starting indices, aka if the job's been parsed up by an mpp handler.
	#
	# maybe make copies? or leave that to the calling function?
	#
	#
	f_denom = float(f_denom or len(Z_fc)+1)
	h_denom = float(h_denom or len(Z_ev)+1)
	#	
	FH=[[0., 0.]]
	if do_sort:
		Z_fc.sort()
		Z_ev.sort()
	n_eq=float(len(Z_ev))
	#n_fc=float(len(Z_fc))
	#
	k_eq = 0
	for j,z_fc in enumerate(Z_fc):	
		#
		if k_eq<n_eq and z_fc>=Z_ev[k_eq]:
			while k_eq<n_eq and z_fc>=Z_ev[k_eq]:
				#
				#FH += [[j+1,z_fc, k_eq+1, Z_ev[k_eq]]]
				#
				#FH += [[(j+1 + j_fc0)/f_denom, (k_eq + j_eq0 +1)/h_denom]]
				#print('ev: ', k_eq, Z_ev[k_eq], z_fc)
				k_eq+=1
			#
			FH += [[(j+1 + j_fc0)/f_denom, (k_eq + j_eq0 )/h_denom]]
		#
	# and one more at the end, just to square off the curve:
	#FH += [[1.,1.]]
	#
	#return [[0 for _ in FH[0]]] + FH
	return FH
	

def roc_test(fignum=1, n_cpu=None):
	n_cpu = (n_cpu or mpp.cpu_count() )
	Z_fc = list(range(10,35))
	Z_eq = list(range(5,25,5))
	Z_eq +=[7,7,7, 20,21,22]
	#Z_eq.sort()
	#Z_fc.sort()
	#
	print('Z_fc: ', Z_fc)
	print('Z_eq: ', Z_eq)
	#
	h_denom=len(Z_eq)
	f_denom=len(Z_fc)
	#
	FH_mpp = calc_roc_mpp(Z_fc, Z_eq, h_denom=h_denom, f_denom=f_denom)
	FH = calc_roc(Z_fc, Z_eq, h_denom=h_denom, f_denom=f_denom)
	#FH_mpp = calc_roc_mpp(Z_fc, Z_eq, h_denom=1, f_denom=1)
	#FH = calc_roc(Z_fc, Z_eq, h_denom=1, f_denom=1)
	#print('FH: ', FH)
	#
	if fignum is not None:
		fg=plt.figure(fignum)
		plt.clf()
		ax1=fg.add_axes([.1,.1,.4,.8])
		ax2=fg.add_axes([.55,.1,.4,.8])
		#
		ax1.plot(*zip(*[[j,k] for j,k in FH]), '-', marker='o')
		ax1.plot(range(2), range(2), ls='-', color='r', lw=2.)
		
		ax1.plot(*zip(*[[j+.0001,k] for j,k in FH_mpp]), '-', marker='o')
		ax2.plot(*zip(*[[j,k] for j,k in FH_mpp]), '-', marker='o')
		ax2.plot(range(2), range(2), ls='-', color='r', lw=2.)
	#
	return FH

def roc_bench(N=1e5):
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