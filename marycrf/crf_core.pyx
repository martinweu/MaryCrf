cimport cython

import numpy as np
from scipy.optimize import minimize


cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from numpy.math cimport INFINITY


cdef double logadd(double a, double b):
    if(a == -INFINITY and b == -INFINITY):
        return -INFINITY;
    elif(a > b):
        return a + log(1+exp(b-a))
    else:
        return b + log(1+exp(a-b))
    
cdef double calc_logZ(np.ndarray[np.double_t, ndim=2]  alphas):
    cdef double z = -INFINITY;
    cdef int i1,i2;
    i1 = alphas.shape[0] -1;
    for i2 in range(alphas.shape[1]):
        z = logadd(z,alphas[i1,i2]);
    return z


cdef class CyCRF:
    
    def __init__(self, long states, FeatureFunction[:] ff):
        self.ff = ff;
        self.ff_len = len(ff);
        self.pi_len = states;

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef double __sumFF(self, double[:] weights, long state_from, long state_to, double[:,:] observations, long t):
        cdef long i = 0;
        cdef double result = 0.0;
        cdef FeatureFunction f;
        
        for i in range(self.ff_len):
            f = self.ff[i] 
            result += f.evaluate(state_from,state_to, observations, t) * weights[i];
        
        return result;
    
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef double __seqsumFF(self,double[:] weights, long[:] state_sequence, double[:,:] observations):
        cdef double fsum;
        cdef long t,i;
        cdef FeatureFunction f;
        fsum = 0;
        
        # Likelihood
        for t in range(observations.shape[1]) :
            for i in range(self.ff_len):
                f = self.ff[i]
                fsum += f.evaluate(state_sequence[t],state_sequence[t+1], observations,t) * weights[i];
        
        return fsum;
    
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __seqvectorFF(self, long[:] state_sequence, double[:,:] observations):
        cdef np.ndarray[np.double_t, ndim=1] fsum;
        cdef long t,i;
        cdef FeatureFunction f;
        fsum = np.zeros((self.ff_len),dtype=np.double);
        
        # Likelihood
        for t in range(observations.shape[1]) :
            for i in range(self.ff_len):
                f = self.ff[i]
                fsum[i] += f.evaluate(state_sequence[t],state_sequence[t+1], observations,t);
        
        return fsum;
        

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __vectorFF(self, long state_from, long state_to, double[:,:] observations, long t):
        cdef long i = 0;
        cdef np.ndarray[np.double_t, ndim=1] result;
        cdef FeatureFunction f;

        result = np.zeros((self.ff_len),dtype=np.double)
        
        for i in range(self.ff_len):
            f = self.ff[i]
            result[i] = f.evaluate(state_from,state_to, observations, t);
        
        return result;
        
    @cython.boundscheck(False)
    cpdef double __evaluateFF(self, double[:] weights, int state_from, int state_to, double[:,:] observations, int t):
        return exp(self.__sumFF(weights,state_from,state_to,observations,t));
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] forward_Log(self, double[:, :] observations, int normalize):
#         runs the forward algorithm in log-space
#     
#         Parameters
#         ----------
#         observations: 2D-array [time,observationIndex]
#         normalize: int 1 for yes ever other value for no 
#                 
#         Returns
#         -------
#         observations: 2D-array [time,stateIndex] forward message  in log space  (either normalized or not)    

        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights);
        return self.__forward_Log_Weights(weights,observations,normalize);
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] __forward_Log_Weights(self, double[:] weights, double[:, :] observations, int normalize):
        cdef np.ndarray[np.double_t, ndim=2] f;
        cdef double acc;
        cdef int t,j,i;
                                                     
        f = np.zeros((observations.shape[1]+1,self.pi_len),dtype=np.double)
        f[0] = np.log(self.pi)

        for t in range(observations.shape[1]):
            for j in range(self.pi_len):
                acc = -INFINITY
                for i in range(self.pi_len):
                    acc = logadd(acc, self.__sumFF(weights,i,j, observations,t) + f[t,i])
                f[t+1,j] = acc
        if normalize == 1 :
            f = np.exp(f) / np.sum(np.exp(f),1)[:,np.newaxis];
        return f
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] forward(self, double[:, :] observations, int normalize):
#         runs the forward algorithm
#     
#         Parameters
#         ----------
#         observations: 2D-array [time,observationIndex]
#         normalize: int 1 for yes ever other value for no 
#                 
#         Returns
#         -------
#         observations: 2D-array [time,stateIndex] forward message (either normalized or not)    
    
        cdef np.ndarray[np.double_t, ndim=2] f;
        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights);
        cdef int t,j,i;
                                                      
        f = np.zeros((observations.shape[1]+1,self.pi_len),dtype=np.double)
        f[0] = self.pi
 
        for t in range(observations.shape[1]):
            for j in range(self.pi_len):
                for i in range(self.pi_len):
                    f[t+1,j] += self.__evaluateFF(weights,i,j, observations,t) * f[t,i]        
        if normalize == 1 :
            f = f / np.sum(f,1)[:,np.newaxis];
        return f
   
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] backward_Log(self, double[:, :] observations):
        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights);
        return self.__backward_Log_Weights(weights,observations); 
   
    @cython.boundscheck(False)
    cdef np.ndarray[np.double_t, ndim=2] __backward_Log_Weights(self, double[:] weights, double[:, :] observations):
        cdef np.ndarray[np.double_t, ndim=2] b;
        cdef double acc;
        cdef int t,j,i;
        
        b = np.zeros((observations.shape[1]+1,self.pi_len),dtype=np.double)

        for t in reversed(range(0,observations.shape[1])):
            for j in range(self.pi_len):
                acc = -INFINITY
                for i in range(self.pi_len):
                    acc = logadd(acc, self.__sumFF(weights,j,i, observations,t) + b[t+1,i])
                b[t,j] = acc
        return b
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] backward(self, double[:, :] observations):
        cdef np.ndarray[np.double_t, ndim=2] b;
        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights);
        cdef int t,j,i;
        
        b = np.zeros((observations.shape[1]+1,self.pi_len),dtype=np.double);
        b[observations.shape[1]] = np.ones(self.pi_len,dtype=np.double) * 10000000;
 
        for t in reversed(range(1,observations.shape[1]+1)):
            for i in range(self.pi_len):
                for j in range(self.pi_len):
                    b[t-1,i] +=  b[t,j] * self.__evaluateFF(weights,i,j, observations,t-1)
 
        return b;
    

    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] forward_backward_Log(self, double[:, :] observations, int normaliz):
#         runs the forward backward algorithm in log-space
#     
#         Parameters
#         ----------
#         observations: 2D-array [time,observationIndex] 
#                 
#         Returns
#         -------
#         observations: 2D-array [time,stateIndex] forward backward message in log space  
    
        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights);
        return self.__forward_backward_Log_Weights(weights,observations,normaliz);

    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] __forward_backward_Log_Weights(self, double[:] weights, double[:, :] observations, int normaliz):
        cdef np.ndarray[np.double_t, ndim=2] b,f,post

        f = self.__forward_Log_Weights(weights,observations,0)
        b = self.__backward_Log_Weights(weights,observations)

        post = f + b;
        if normaliz == 1:
            post = np.exp(post) / np.sum(np.exp(post),1)[:,np.newaxis];
        
        return post;
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=2] forward_backward(self, double[:, :] observations):
#        runs the forward backward algorithm
#    
#        Parameters
#        ----------
#        observations: 2D-array [time,observationIndex] 
#                
#        Returns
#        -------
#        observations: 2D-array [time,stateIndex] forward backward message
#    
        cdef np.ndarray[np.double_t, ndim=2] b,f,post
 
        f = self.forward(observations,0)
        b = self.backward(observations)
 
        post = f * b;
 
        post = post / np.sum(post,1)[:,np.newaxis]
        return post;
   
    @cython.boundscheck(False)
    cpdef tuple viterbi(self, double[:, :] observations):
#        runs the viterbi algorithm
#    
#        Parameters
#        ----------
#        observations: 2D-array [time,observationIndex] 
#                
#        Returns
#        -------
#        viterbi-path: 1D-array the viterbi path
#        viterbi-message: 2D-array the viterbi message

        cdef np.ndarray[np.double_t, ndim=1] weights = np.asarray(self.weights)
        return self.__viterbi_Weights(weights,observations);
    
    @cython.boundscheck(False)
    cpdef tuple __viterbi_Weights(self, double[:] weights, double[:, :] observations):
        cdef np.ndarray[np.double_t, ndim=2] v
        cdef np.ndarray[np.double_t, ndim=1] tmp
        cdef np.ndarray[np.int_t, ndim=2] bp
        cdef np.ndarray[np.int_t, ndim=1] path
        cdef int t,j,i
                                                     
        v = np.zeros((observations.shape[1]+1,self.pi_len),dtype=np.double)
        bp = np.zeros((observations.shape[1],self.pi_len),dtype=np.int)
        path = np.zeros((observations.shape[1]+1),dtype=np.int)
        tmp = np.zeros((self.pi_len),dtype=np.double)

        v[0] = self.pi

        for t in range(observations.shape[1]):
            for j in range(self.pi_len):
                for i in range(self.pi_len):
                    tmp[i] = self.__evaluateFF(weights,i,j, observations,t) * v[t,i]
                v[t+1,j] = tmp.max()
                bp[t,j] = np.argmax(tmp)        
            v[t+1] = v[t+1] / v[t+1].sum()
        #v = v / v.sum(1)[:,np.newaxis]
    
        t = observations.shape[1]
        path[t] = np.argmax(v[t])
        
        while t > 0:
            path[t-1] = bp[t-1,path[t]]
            t -= 1
        
        return path,v

    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __gradient(self, double[:] weights, list state_sequences, list observations):
#         calculates the gradient of the cost-function for the point specified by weights
#       
#         the state sequence must have the same list index as the corresponding observation sequence
#     
#         Parameters
#         ----------
#         weights: 1D-array weights for the feature functions
#         state_sequences: the state sequences used for training
#         observations:    the observations used for training
#         
#                 
#         Returns
#         -------
#         gradient: 1D-array the gradients at the point weights
    
        cdef np.ndarray[np.double_t, ndim=1] result
        cdef np.ndarray[np.double_t, ndim=1] tmp
        cdef np.ndarray[np.double_t, ndim=2] alpha
        cdef np.ndarray[np.double_t, ndim=2] beta
        cdef int t,i,j,k
        cdef double sigma = 10.0
        cdef double z
        result = np.zeros((self.ff_len),dtype=np.double);
        
    
        for k in range(len(state_sequences)):
            alpha = self.__forward_Log_Weights(weights,observations[k],0);
            beta = self.__backward_Log_Weights(weights,observations[k]);
            z = calc_logZ(alpha);

            result += self.__seqvectorFF(state_sequences[k],observations[k]);
    
        # Likelihood-Normalization
            for t in range(0,observations[k].shape[1]) :
                for i in range(self.pi_len) :        
                    for j in range(self.pi_len) :
                        tmp = self.__vectorFF(j,i, observations[k],t);
                        result -= exp(alpha[t,j] + np.sum(weights*tmp) + beta[t+1,i] -z) * tmp        
        
        # Regularization
        result -= np.array(weights,dtype=np.double)/(sigma*sigma);
        
        # Negative likelihood because of minimum search
        return -result;

    @cython.boundscheck(False)
    cpdef double __logLikelihood(self,double[:] weights, list state_sequences, list observations):
#         calculates the log likelihood for the point specified by weights
#       
#         the state sequence must have the same list index as the corresponding observation sequence
#     
#         Parameters
#         ----------
#         weights: 1D-array weights for the feature functions
#         state_sequences: the state sequences used for training
#         observations:    the observations used for training
#         
#                 
#         Returns
#         -------
#         log likelihood: double the log likelihood at the point weights
    
        cdef int i
        cdef double sigma = 10.0
        cdef double result
        result = 0;
        
        fsum = np.zeros((self.ff_len),dtype=np.double);
        
        for i in range(len(state_sequences)):
            result += self.__seqsumFF(weights,state_sequences[i],observations[i]);
            result -= calc_logZ(self.__forward_Log_Weights(weights,observations[i],0))
            
        # Likelihood-Normalization
        
        
        #result += np.sum(weights * fsum);
        # Regularization
        result -= np.sum(np.power(weights,2))/(2*sigma*sigma);
        
        # Negative likelihood because of minimum search
        return -result;
        
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __optimal_weights_NM(self, list state_sequences, list observations, np.ndarray[np.double_t, ndim=1] ini,int it):
        return minimize(self.__logLikelihood,x0=ini,args=(state_sequences,observations), method='Nelder-Mead', options={'maxfev': it}).x; 
    
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __optimal_weights_BFGS(self, list state_sequences, list observations, np.ndarray[np.double_t, ndim=1] ini):
        return minimize(self.__logLikelihood,x0=ini,jac=self.__gradient,args=(state_sequences,observations), method='BFGS', options={'gtol': 1e-6, 'disp': False}).x 

    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __generate_initial_weights(self):
        return np.log(abs(np.random.normal(0.5, 0.25, self.ff_len)));
    
    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] __calc_Pi(self, list state_sequences):
        cdef np.ndarray[np.double_t, ndim=1] result;
        cdef long[:] state_sequence;
        result = np.zeros(self.pi_len,dtype=np.double);
        for state_sequence in state_sequences:
            for i in state_sequence:
                result[i] += 1;
        
        result /= result.sum()
        return result;

    
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] train_NM(self, list state_sequences, list observations,int it):
        cdef np.ndarray[np.double_t, ndim=1] result
        self.pi = self.__calc_Pi(state_sequences);
        result = self.__optimal_weights_NM(state_sequences,observations,self.__generate_initial_weights(),it);
        self.weights = result; 
        return result;
    
        
    @cython.boundscheck(False)
    cpdef np.ndarray[np.double_t, ndim=1] train_BFGS(self, list state_sequences, list observations):
        cdef np.ndarray[np.double_t, ndim=1] result
        self.pi = self.__calc_Pi(state_sequences);
        result = self.__optimal_weights_BFGS(state_sequences,observations,self.__generate_initial_weights());
        self.weights = result; 
        return result;


    
cdef class FeatureFunction(object):
    cdef public double evaluate(self, long y_from, long y_to, double[:, :] x_seq, long t):
        return self.calc(y_from, y_to, x_seq, t);
    
    cdef double calc(self, long y_from, long y_to, double[:, :] x_seq, long t):
        return 0.0;

