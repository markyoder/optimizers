import glob
import os
import imp

excluded_py_files=['__init__.py']
local_path = os.path.split(__file__)[0]

__all__ = [os.path.splitext(os.path.split(g)[-1])[0] for g in glob.glob(os.path.join(local_path,'*.py')) if not os.path.split(g)[-1] in excluded_py_files]

for ky in __all__:
	#mod = 'optimizers.{}'.format(ky)
	#
	#print('importing: %s' % ky)
	#locals()[ky]=__import__(mod)
	# i think this is a stupid way to do this, but it's the only way that's working. probably, there is a smart way to use __import__(),
	# but i haven't found it...
	exec('from optimizers import %s' % ky)

