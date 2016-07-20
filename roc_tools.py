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


def calc_roc(Z_fc, Z_ev, f_denom=None, h_denom=None):
	# (we should figure out how to compile these with numba)l.\
	#
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
	#
	# eventually, we might reinstate this model, but i think that we don't need it if we're stepping through the list(s) from the top down.
	#
	#it_ev =enumerate(reversed(sorted([(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev], key=lambda rw: rw[0])))
	#for rw in it_ev: print(rw)
	#it_ev =enumerate(reversed(sorted([(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev], key=lambda rw: rw[0])))
	it_ev =enumerate(reversed(sorted(Z_ev)))
	
	N_fc = len(Z_fc)
	N_ev = len(Z_ev)
	#
	k_fc_max = len(Z_fc)
	#k_ev, (z_ev, n_ev) = it_ev.__next__()		# eventually trap for the case with no events.
	k_ev, z_ev = it_ev.__next__()		# eventually trap for the case with no events.
	

	# explicitly declare the iterator so we can manipulate it directly:
	it_fc = enumerate(reversed(sorted(Z_fc)))
	for k_fc,z_fc in it_fc:
		
		#
		#
		if k_ev<N_ev and z_ev>=z_fc:
			#while k_ev<len(Z_events) and Z_events[k_ev][0]>=z_fc:
			while k_ev<len(Z_ev) and z_ev>=z_fc:
				#H += n_ev
				H += 1
				#H+=Z_events[k_ev][1]
				#k_ev+=1
				#print('** ', z_ev, z_fc, k_ev)
				#print('advancing events..', k_ev, z_ev)
				if k_ev==N_ev-1:
					k_ev+=1
					break
				#k_ev, (z_ev, n_ev) = it_ev.__next__()
				k_ev, z_ev = it_ev.__next__()
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

def calc_roc_compressed(Z_fc, Z_ev, f_denom=None, h_denom=None, do_sort=True,  do_compressed=True):
	# A minor variaiton on calc_roc() with compression, aka: we exclude intermediate points in segments.
	# the loop-exits are probably still a bit sloppy. unfortunately, i don't think there is an elegant way to exit an iterator; you have to either
	# count rows or trap an exception.
	#
	# this almost works, but not quite. i think there is a counting/logic error when the first forecast sites don't predict an event... or something.
	#
	# first, force Z_ev to be an array of arrays; if len=1
	F,H = 0,0
	FH = [[F,H]]
	# we can do this more efficiently with an iterator...
	#
	# get an iterator for the events data:
	it_ev =enumerate(reversed(sorted([(numpy.append(numpy.atleast_1d(x), [1]))[0:2] for x in Z_ev], key=lambda rw: rw[0])))
	
	N_fc = len(Z_fc)
	N_ev = len(Z_ev)
	#
	k_fc_max = len(Z_fc)
	k_ev, (z_ev, n_ev) = it_ev.__next__()		# eventually trap for the case with no events.
	
	   
	# explicitly declare the iterator so we can manipulate it directly:
	it_fc = enumerate(reversed(sorted(Z_fc)))
	k_fc,z_fc = it_fc.__next__()
	#for k_fc,z_fc in it_fc:
	while k_fc<N_fc:
		#
		if k_ev<N_ev and z_ev>=z_fc:
			#while k_ev<len(Z_events) and Z_events[k_ev][0]>=z_fc:
			while k_ev<len(Z_ev) and z_ev>=z_fc:
				H += n_ev
				if k_ev==N_ev-1:
					k_ev+=1
					break
				k_ev, (z_ev, n_ev) = it_ev.__next__()
			#
			if k_ev<N_ev:
				k_fc,z_fc = it_fc.__next__()
			else:
				k_fc+=1
		else:
			if do_compressed:
				while k_fc<N_fc and z_ev<z_fc:
					F+=1
					if k_fc == N_fc-1:
						k_fc+=1
						break
					k_fc,z_fc = it_fc.__next__()
			else:
				F+=1
				if k_fc == N_fc-1:
					k_fc+=1
					break
				k_fc,z_fc = it_fc.__next__()
					
		FH += [[F,H]]
	#
	f_denom = (f_denom or max([rw[0] for rw in FH]))
	h_denom = (h_denom or max([rw[1] for rw in FH]))
	for rw in FH:
		rw[0]/=f_denom
		rw[1]/=h_denom
	
	return FH
#
# eventually, we'll want to fold this (and other bits) into some sort of ROC class, i think, but for now it's procedural...
class ROC_data_handler(object):
	def __init__(self, fc_xyz, events_xyz=None, dx=None, dy=None, fignum=0, do_clf=True, z_event_min=None, z_events_as_dicts=False):
		#
		# get roc Z_fc, Z_ev from an xyz format forecast and test-catalog object.
		# for now, assume lattice sites are center-binned.
		# ... and eventually break this up
		# ... be careful with variable declaration, so we can (maybe) just load all the locals() into self.__dict__
		#
		if isinstance(fc_xyz, str):
			# if we're given a filename...
			with open(fc_xyz, 'r') as froc:
				fc_xyz= [[float(x) for x in rw.split()] for rw in froc if rw[0] not in('#', ' ', '\t', '\n')]
		if isinstance(events_xyz, str):
			# we're given a filename. load it ip:
			with open(events_xyz,'r') as fev:
				events_xyz = [[float(x) for x in rw.split()] for rw in fev if rw[0] not in('#', ' ', '\t', '\n')]
		#
		if not hasattr(fc_xyz, 'dtype'):
			fc_xyz = numpy.core.records.fromarrays(zip(*fc_xyz), names=('x','y','z'), formats=['>f8', '>f8', '>f8'])
		if not hasattr(events_xyz, 'dtype'):
			events_xyz = numpy.core.records.fromarrays(zip(*events_xyz), names=('x','y','z'), formats=['>f8', '>f8', '>f8'])
		z_event_min = (z_event_min or min(events_xyz['z']))
		#
		y_range = [min(fc_xyz['y']), max(fc_xyz['y'])]
		x_range = [min(fc_xyz['x']), max(fc_xyz['x'])]
		#
		X_set = sorted(list(set(fc_xyz['x'])))
		Y_set = sorted(list(set(fc_xyz['y'])))
		d_x = (dx or abs(X_set[1] - X_set[0]))
		d_y = (dy or abs(Y_set[1] - Y_set[0]))
		#
		nx = len(X_set)
		ny = len(Y_set)
		#print('nx,ny: ', nx, ny)
		#
		self.__dict__.update({key:val for key,val in locals().items() if not key in ('rw', 'x','y','z','fc_index', 'self')})
		#
		# ... so eventually wite this up as a class object...
		# note the center-binning default (x - (lon_0-bin_width)), ...
		#get_site = lambda x,y: int(round((x-lons[0]+.5*d_lon)/d_lon)) + int(round((y-lats[0]+.5*d_lat)/d_lat))*nx
		#
		# now, get the event z-values. what we really want is sites with events (and their z-values), so we need to keep track of the bin index of that site
		# and the number of events in that bin.
		if z_events_as_dicts:
			z_events = {}
			for x,y,z in events_xyz:
				if z<z_event_min: continue
				# is this event in bounds?
				if x>self.x_max or x<self.x_min or y>self.y_max or y<self.y_min: continue
				#
				fc_index = self.get_site_index(x,y)
				if not fc_index in z_events.keys(): z_events[fc_index]=[fc_xyz[fc_index][2],0]
				z_events[fc_index][1] +=1
			#
		else:
			z_events = [fc_xyz[self.get_site_index(x,y)]['z'] for x,y,z in events_xyz if not (x>self.x_max or x<self.x_min or y>self.y_max or y<self.y_min) and z>=z_event_min]
		#
		self.z_events = reversed(sorted(z_events))
		self.z_fc = reversed(sorted(fc_xyz['z']))
		#
	def get_site_index(self,x,y):
		#return int(round((x-self.x_range[0]+.5*self.d_x)/self.d_x)) + int(round((y-self.y_range[0]+.5*self.d_y)/self.d_y))*self.nx
		#print('*** ', x, y, int((x-self.x_range[0]+.5*self.d_x)/self.d_x) + int((y-self.y_range[0]+.5*self.d_y)/self.d_y)*int(self.nx))
		return int((x-self.x_range[0]+.5*self.d_x)/self.d_x) + int((y-self.y_range[0]+.5*self.d_y)/self.d_y)*int(self.nx)
	#
	def calc_roc(self):
		# we'll need to know if z_events is in a list or dict format. for the time being, i think we're not going to support the dict model
		# ( Z_ev = {j:[z,n] ,...] ) that we developed earlier. memory can be premium for global forcasts, so for now, if we have a dict, expand it.
		# ... or just save both formats for now...
		z_ev = self.Z_events_dict_to_list(self.z_events)
		#
		return calc_roc(Z_fc=self.z_fc, Z_ev=self.z_events, f_denom=None, h_denom=None)
		
	def Z_events_dict_to_list(self, z_dict=None):
		if z_dict is None: z_dict=self.z_events
		if not isinstance(z_dict,dict): return z_dict
		#
		# convert {j:[z,n], ...} --> [z, ...]
		z_out = []
		for key,(x,n) in z_dict.items():
			for j in range(n):
				z_out += [x]
			#
		#
		z_out.sort()
		return return z_out
	#
	# some convenience functions
	@property
	def x_min(self):
		return self.x_range[0] - .5*self.d_x
	@property
	def x_max(self):
		return self.x_range[1] + .5*self.d_x
	@property
	def y_min(self):
		return self.y_range[0] - .5*self.d_y
	@property
	def y_max(self):
		return self.y_range[1] + .5*self.d_y
	
#
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
