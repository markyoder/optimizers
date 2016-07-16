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
# for unit testing:
default_roc_sample = {'F': [17,17,16,15,14,13,12,11,10,9,8,8,7,6,5,4,4,3,3,2,1,1,0,0,0], 'H':[8,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,5,5,4,4,4,3,3,2,1], 'Z_fc':list(range(10,35)), 'Z_ev':[10., 20., 25, 27, 30, 32, 33, 34]}

#
def calc_roc_mpp(Z_fc, Z_ev, n_cpu=None, f_denom=None, h_denom=None):
	# mpp handler for roc.
	# TODO: ok, so this looks like it's working, but a more quantitative test with larget data sets is in order. first, do a point-by-point on the spp roc; then bench mpp vs spp
	# on a large array... also, add a "stepify" function, so spp and mpp curves can be explicitly compared... or write a script to parse out redundant elements (if x[0][0]==x[1][0]
	# or x[0][1]==x[1][1], remove the later one). we get these elements because one sub-process will find the "hit" at a different z_fc than another process.
	#
	n_cpu = (n_cpu or mpp.cpu_count())
	#
	# there should probably be recipricol "if" hadling in calc_roc() (aka, allow an n_cpu input which diverts here).
	if n_cpu==1:
		return calc_roc(Z_fc, Z_ev, f_denom=f_denom, h_denom=h_denom)
	#
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

def calc_roc(Z_fc, Z_ev, f_denom=None, h_denom=None, j_fc0=0, j_eq0=0, do_sort=True, n_cpu=1):
	# let's take an iterative approach, and also accomodate cases where multiple events occur in a single bin... sometimes.
	# this reqires that we think of Z_ev not as "the z-values for events," but "the z-values for cells that have events, and we want to know
	# how many events.", so Z_ev --> [[j,z,n], ...] where j-> index (which we don't really need), z-> value, n-> number of events).
	# for now, ignore the index? we could keep the index, and then we can explicitly check it against the forecast values and not increment F
	# if j1=j2... which in general may be another way to do this... but more on that later.
	# if n is omitted, we assume n=1.
	#
	# also, as a general approach, let's start with (F,H) = (0,0) (aka, highest values) and increment backwards. for each step, H,F can move either
	# up (H+=1) or to the right (F+=1). in other words, for each new exposed site, we either have a hit or a miss (false alarm).
	
	#
	# first, force Z_ev to be an array of arrays; if len=1
	F,H = 0,0
	FH = [[F,H]]
	# we can do this more efficiently with an iterator...
	
	#it_ev =enumerate(reversed(sorted([(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev], key=lambda rw: rw[0])))
	#for rw in it_ev: print(rw)
	it_ev =enumerate(reversed(sorted([(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev], key=lambda rw: rw[0])))
	
	N_fc = len(Z_fc)
	N_ev = len(Z_ev)
	
	#
	#Z_events = [(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev]
	#Z_events.sort(key=lambda rw: rw[0])
	#Z_events.reverse()
	#
	#Z_forecast = Z_fc		# we might end up making a copy of this (or something).
	#Z_forecast = reversed(sorted(Z_fc))
	#
	#k_ev = 0
	#k_fc = 0
	k_fc_max = len(Z_fc)
	k_ev, (z_ev, n_ev) = it_ev.__next__()		# eventually trap for the case with no events.
	

	# explicitly declare the iterator so we can manipulate it directly:
	it_fc = enumerate(reversed(sorted(Z_fc)))
	for k_fc,z_fc in it_fc:
		
		#
		#
		if k_ev<len(Z_ev) and z_ev>=z_fc:
			#while k_ev<len(Z_events) and Z_events[k_ev][0]>=z_fc:
			while k_ev<len(Z_ev) and z_ev>=z_fc:
				H += n_ev
				#H+=Z_events[k_ev][1]
				#k_ev+=1
				print('** ', z_ev, z_fc, k_ev)
				#print('advancing events..', k_ev, z_ev)
				if k_ev==N_ev-1:
					k_ev+=1
					break
				k_ev, (z_ev, n_ev) = it_ev.__next__()
			#
		else:
			F+=1
			# looping this way adds extra iterations. can we reorganize to "while" through repeteating F-steps as well?
			#while z_ev<z_fc:
			#	F+=1
			#	if k_fc == N_fc-1:
			#		k_fc+=1
			#		break
			#	k_fc,z_fc = it_fc.__next__()

		
		FH += [[F,H]]
	#
	f_denom = (f_denom or max([rw[0] for rw in FH]))
	h_denom = (h_denom or max([rw[1] for rw in FH]))
	for rw in FH:
		rw[0]/=f_denom
		rw[1]/=h_denom
	
	return FH
	
	



def roc_test_spp1(fignum=1):
	#
	# code up an explicit ROC input and output.
	# keep it small and simple, with just enough twists to be rigorous.
	# assume sparse approximations, so no two events with the same value.
	#Z_fc = list(range(10,35))
	#Z_ev = [10., 20., 25, 27, 30, 32, 33, 34]
	#
	# work out the proper shift; this is a > vs >= type thing.
	#F = [17,16,15,14,13,12,11,10, 9,8,8,7,6,5,4,4,3,3,2,1,1,0,0,0,0]
	#F = [17,17,16,15,14,13,12,11,10,9,8,8,7,6,5,4,4,3,3,2,1,1,0,0,0]   # inclusive: it's a falsie if z>=z0 and empty.  (this is probably correct)
	#H = [8,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,5,5,4,4,4,3,3,2,1]
	#
	Z_fc, Z_ev, F,H = (default_roc_sample[x] for x in ('Z_fc', 'Z_ev', 'F','H'))
	#
	n_ev = max(H)
	n_fc = max(F)
	#n_ev=1
	#n_fc=1
	#
	plt.figure(fignum)
	plt.clf()
	#
	plt.plot(numpy.array(F)/n_fc, numpy.array(H)/n_ev, '.-')
	plt.plot(range(2), range(2), 'r-', lw=2)
	plt.gca().set_ylim(-.2, max(H)*1.1/n_ev)
	plt.gca().set_xlim(-.2, max(F)*1.1/n_fc)
	
	#
	print('lens: ', len(F), len(H))	

def roc_test(fignum=1, n_cpu=None):
	# ... so this still isn't quite right.
	n_cpu = (n_cpu or mpp.cpu_count() )
	#Z_fc = list(range(10,35))
	#Z_ev = list(range(5,25,5))
	Z_fc = list(range(1,26))
	#Z_ev = list(range(20,26))
	Z_ev = [25,25,25,25]
	
	#Z_ev +=[7,7,7, 20,21,22]
	#Z_ev.sort()
	#Z_fc.sort()
	#
	print('Z_fc: ', Z_fc)
	print('Z_ev: ', Z_ev)
	#
	h_denom=len(Z_ev)
	f_denom=len(Z_fc)
	#
	FH_mpp = calc_roc_mpp(Z_fc, Z_ev, h_denom=h_denom, f_denom=f_denom)
	FH = calc_roc(Z_fc, Z_ev, h_denom=h_denom, f_denom=f_denom)
	#FH_mpp = calc_roc_mpp(Z_fc, Z_ev, h_denom=1, f_denom=1)
	#FH = calc_roc(Z_fc, Z_ev, h_denom=1, f_denom=1)
	#print('FH: ', FH)
	#
	if fignum is not None:
		fg=plt.figure(fignum)
		plt.clf()
		ax1=fg.add_axes([.1,.1,.4,.8])
		ax2=fg.add_axes([.55,.1,.4,.8])
		#
		ax1.plot(*zip(*[[j,k] for j,k in FH]), '-', marker='o', label='spp')
		ax1.plot(range(2), range(2), ls='-', color='r', lw=2.)
		
		ax1.plot(*zip(*[[j+.01,k] for j,k in FH_mpp]), '-', marker='o', label='mpp (shifted)')
		ax2.plot(*zip(*[[j,k] for j,k in FH_mpp]), '-', marker='o', label='mpp')
		ax2.plot(range(2), range(2), ls='-', color='r', lw=2.)
		#
		plt.suptitle('ROC spp and mpp comparison', size=14)
		ax1.set_title('ROC_spp', size=14)
		ax2.set_title('ROC_mpp', size=14)
		for ax in (ax1,ax2):
			ax.legend(loc=0, numpoints=1)
		#
	#
	return FH

def roc_test_fig(N_ev=1000, N_fc=10000, fignum=0, do_clf=True, n_cpu=1):
	N_ev=int(N_ev)
	N_fc=int(N_fc)
	R1=random.Random()
	R2=random.Random()
	#
	Z_fc = [R1.random() for _ in range(N_fc)]
	Z_ev = [R2.random() for _ in range(N_ev)]
	#
	# we can actulaly let calc_roc_mpp() or maybe calc_roc() handle this now... if we want.
	FH = calc_roc(Z_fc=Z_fc, Z_ev=Z_ev, n_cpu=n_cpu)
	#if n_cpu==1:
	#	FH = calc_roc(Z_fc=Z_fc, Z_ev=Z_ev)
	#else:
	#	FH = calc_roc_mpp(Z_fc=Z_fc, Z_ev=Z_ev, n_cpu=n_cpu)
	#
	plt.figure(fignum)
	if do_clf:
		plt.clf()
		plt.plot(range(2), range(2), ls='-', lw=2., color='r')
	plt.plot(*zip(*FH), marker='.')
	#
	return FH
	#

def roc_bench(N=1e5):
	N=int(N)
	R1=random.Random()
	R2=random.Random()
	#
	Z_fc = [R1.random() for _ in range(N)]
	Z_ev = [R2.random() for _ in range(int(.1*N))]
	#
	print('bench for spp and mpp.')
	#
	# first make sure roc is compiled:
	FH=calc_roc(list(range(5)), list(range(5)))
	#
	t0 = time.time()
	for k in range(100):
		FH = calc_roc(Z_fc, Z_ev)
	#
	print('dt_2: ', time.time()-t0)
	#
	t0 = time.time()
	for k in range(100):
		FH = calc_roc_mpp(Z_fc, Z_ev)
	#
	print('dt: ', time.time()-t0)

if __name__=='__main__':
	pass
else:
	plt.ion()
#
