import numpy as np
from scipy.optimize import minimize


cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.math cimport log



cdef class CyCRF:
    
    cdef public np.double_t[:] pi
    cdef public np.double_t[:] weights
    cdef public FeatureFunction[:] ff;
    cdef public long ff_len;
    cdef public long pi_len;

    cpdef double __sumFF(self, double[:] weights, long state_from, long state_to, double[:,:] observations, long t)
    
    cpdef double __seqsumFF(self,double[:] weights, long[:] state_sequence, double[:,:] observations)

    cpdef np.ndarray[np.double_t, ndim=1] __seqvectorFF(self, long[:] state_sequence, double[:,:] observations)
        
    cpdef np.ndarray[np.double_t, ndim=1] __vectorFF(self, long state_from, long state_to, double[:,:] observations, long t)
        
    cpdef double __evaluateFF(self, double[:] weights, int state_from, int state_to, double[:,:] observations, int t)
    
    cpdef np.ndarray[np.double_t, ndim=2] forward_Log(self, double[:, :] observations, int normalize)
    cpdef np.ndarray[np.double_t, ndim=2] __forward_Log_Weights(self, double[:] weights, double[:, :] observations, int normalize)
    
    cpdef np.ndarray[np.double_t, ndim=2] forward(self, double[:, :] observations, int normalize)
   
    cpdef np.ndarray[np.double_t, ndim=2] backward_Log(self, double[:, :] observations)
    cdef np.ndarray[np.double_t, ndim=2] __backward_Log_Weights(self, double[:] weights, double[:, :] observations)
    
    cpdef np.ndarray[np.double_t, ndim=2] backward(self, double[:, :] observations)

    cpdef np.ndarray[np.double_t, ndim=2] forward_backward_Log(self, double[:, :] observations, int normalize)
    cpdef np.ndarray[np.double_t, ndim=2] __forward_backward_Log_Weights(self, double[:] weights, double[:, :] observations, int normalize)
    
    cpdef np.ndarray[np.double_t, ndim=2] forward_backward(self, double[:, :] observations)
   
    cpdef tuple viterbi(self, double[:, :] observations)
    cpdef tuple __viterbi_Weights(self, double[:] weights, double[:, :] observations)

    cpdef np.ndarray[np.double_t, ndim=1] __gradient(self, double[:] weights, list state_sequences, list observations)
    
    cpdef double __logLikelihood(self,double[:] weights, list state_sequences, list observations)
    
    cpdef np.ndarray[np.double_t, ndim=1] __optimal_weights_NM(self, list state_sequences, list observations, np.ndarray[np.double_t, ndim=1] ini,int it)
    
    cpdef np.ndarray[np.double_t, ndim=1] __optimal_weights_BFGS(self, list state_sequences, list observations, np.ndarray[np.double_t, ndim=1] ini)

    cpdef np.ndarray[np.double_t, ndim=1] __generate_initial_weights(self)    
    
    cpdef np.ndarray[np.double_t, ndim=1] __calc_Pi(self, list state_sequences)
    
    cpdef np.ndarray[np.double_t, ndim=1] train_NM(self, list state_sequences, list observations,int it)
    
    cpdef np.ndarray[np.double_t, ndim=1] train_BFGS(self, list state_sequences, list observations)


cdef class FeatureFunction(object):
    cdef public double evaluate(self, long y_from, long y_to, double[:, :] x_seq, long t)
    
    cdef double calc(self, long y_from, long y_to, double[:, :] x_seq, long t)