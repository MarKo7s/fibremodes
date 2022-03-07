# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:31:47 2020

Mode generation class

@author: Marcos

"""
import mode_generation_core_library as mgcl
#import numpy as np # Already imported in mgcl

class LGmodes():
    """
    Creates an object that can contain and create LGmodes:
        Input parameters:
            - mfd = mode field diameter of the funtamental mode
            - group = number of groups
            - N = number of samples in of the axis Ex: 1080 --> 1080x1080 modes
            - px_size = step size 
            - generateModes:{True or False} -- if you want to generate the modes when the object is created
            - wholeSet = {True or False} -- if you want to generate all mode pyramid or half (no complex conj)
            - targetCore = {'GPU' or 'CPU'} 
            - multicore = {true or false} -- CPU computations done in parallel by numba and Ipyrallel if coreTarget is CPU
                    GPU:True uses cupy and numba
                    CPU:True uses Ipyparallel and numba
        Output parameters:
            - index = modes coefs
            - LGmodesArray = Array with the modes
            - LGmodesArray__ = Array with the whole set of modes
            
    """
    
    def __init__(self,mfd,group,N,px_size, generateModes = True, wholeSet = False, engine = 'GPU', multicore = True):
        #Input atributes
        self.w0 = mfd/2
        self.group = group
        self.N = N 
        self.px_size = px_size
        
        #AutoGenerated atributes and constructor
        xx = mgcl.np.arange(-N//2,N//2,1) * px_size
        self.X,self.Y = mgcl.np.meshgrid(xx,xx)     
        self.index = mgcl.np.array([0])
        self.LGmodesArray = 'None'
        self.LGmodesArray__ = 'None'
        self.numModes = 'None'
        self.numModesAll = 'None'
        self.parameters = 0
  
        if generateModes == True:
            self.index = self.computeCoefs()
            self.LGmodesArray = self.computeLGmodes(engine, multicore)
            self.numModes = self.LGmodesArray.shape[0]
            if wholeSet == True:
                self.LGmodesArray__ = self.computeAllmodes(multicore)
                self.numModesAll = self.LGmodesArray__.shape[0]
            
                
        self.parameters = self.getSpecs()          
        
    
        
    def __repr__(self):
        
        return( 'LG_object_mfd:%d_group:%i' %(self.parameters['mfd'],self.parameters['mode_group']) )
      
      
    
    #Methods:
        
    def computeCoefs(self):
        xx = self.group
        print('Generating modes coeficients...')
        return( mgcl.graded_index_fiber_coefs(xx))
        
    def updateCoefs(self):
        """ 
        Update or compute coefs in case you changed object atributes 
        
        """
        self.index = mgcl.graded_index_fiber_coefs(self.group)
        
    def computeLGmodes(self, targetEngine, multi):
        print('Generating modes...')
        return( mgcl.LGmodes(self.w0,self.X,self.Y,self.index, engine = targetEngine, multicore = multi) )
    
    def updateLGmodes(self,targetEngine, multi):
         """ 
         Update or compute LGmodes in case you changed object atributes 
         """
         self.LGmodesArray = mgcl.LGmodes(self.w0,self.X,self.Y,self.index, engine = targetEngine, multicore = multi)
         self.numModes = self.LGmodesArray.shape[0]
    
    def computeAllmodes(self, multi):
        print('Generating rest of the modes...')
        return( mgcl.computeWholeSetofModes( self.LGmodesArray, self.index,multicore = multi ) )
    
    def updateLGmodesAll(self,multi):
         """ 
         Update or compute LGmodes in case you changed object atributes 
         """
         self.LGmodesArray__ = mgcl.computeWholeSetofModes(self.LGmodesArray, self.index,multicore = multi)
         self.numModesAll = self.LGmodesArray__.shape[0]
      
    def getSpecs(self):
        p = {
                      'mfd': (2*self.w0),
                      'mode_group': self.group,
                      'num_samples': self.N,
                      'px_size' : self.px_size,
                      'num_modes' : self.numModes,
                      'num_modes_all' : self.numModesAll
                      }
        self.parameters = p
        return(p)
    
        # print('LG modes specs:')
        
        # for key,value in parameters.items():
        #     print(key, value)
    
        
        
if __name__ == '__main__':
    t = mgcl.times
    import cupy as cp
    print(cp.cuda.Device(0).mem_info)
    t.tic()
    LGgpu = LGmodes(365, 60 , 320, 9.2 , generateModes = True, wholeSet = False, engine = 'GPU', multicore = True)
    t.toc()
    
    #t.tic()
    #LGold = LGmodes(15.3,11,1024,1.661217730978261,generateModes = True, wholeSet = False, engine = 'CPU', multicore = False)
    #t.toc()
    

































