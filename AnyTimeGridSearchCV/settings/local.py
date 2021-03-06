'''
Created on Jan 9, 2018

Settings for running with a scheduler running on localhost (must have scheduler running on same machine)

@author: Ory Jonay
'''

from AnyTimeGridSearchCV.settings.base import *

DASK_SCHEDULER_PARAMS = {'address': ([l 
                                      for l in ([ip 
                                                 for ip in socket.gethostbyname_ex(socket.gethostname())[2] 
                                                 if not ip.startswith("127.")][:1], 
                                                [[(s.connect(('8.8.8.8', 53)), 
                                                   s.getsockname()[0], s.close()) 
                                                  for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) 
                                      if l][0][0])+ ':8786'}
