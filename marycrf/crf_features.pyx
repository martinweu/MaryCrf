cimport cython

cimport crf_core

cdef class FeatureFunctionStateTo(crf_core.FeatureFunction):
    cpdef public long state_to
    def __init__(self, long state_to):
        self.state_to = state_to;
    cdef double calc(self, long state_from, long state_to, double[:, :] observations, long t):
        if state_to == self.state_to:
            return self.calcStateTo(state_from,observations,t);
        return 0.0; 
    cdef double calcStateTo(self, long state_from, double[:, :] observations, long t):
        return 0.0;
        
cdef class FeatureFunctionStateToIndexedCurrentObservation(FeatureFunctionStateTo):
    cpdef public long observationIndex
    def __init__(self, long state_to, long observationIndex):
        super().__init__(state_to);
        self.observationIndex = observationIndex;
        
    cdef double calcStateTo(self, long state_from, double[:, :] observations, long t):
        return self.calcStateToIndexedCurrentObservation(state_from,observations[self.observationIndex,t]);
    
    cdef double calcStateToIndexedCurrentObservation(self, long state_from, double current_observation):
        return 0.0;
        
cdef class FeatureFunctionCurrentStateX(FeatureFunctionStateToIndexedCurrentObservation):
    def __init__(self, long state_to,long observationIndex):
        super().__init__(state_to,observationIndex);
        
    cdef double calcStateToIndexedCurrentObservation(self, long state_from, double current_observation):
        return current_observation;
    
cdef class FeatureFunctionCurrentStateXSq(FeatureFunctionStateToIndexedCurrentObservation):
    def __init__(self, long state_to,long observationIndex):
        super().__init__(state_to,observationIndex);
    cdef double calcStateToIndexedCurrentObservation(self, long state_from, double current_observation):
        return current_observation * current_observation;
    
cdef class FeatureFunctionCurrentStateConst(FeatureFunctionStateTo):
    def __init__(self, long state_to):
        super().__init__(state_to);
    cdef double calcStateTo(self, long state_from, double[:, :] observations, long t):
        return 1.0;

cdef class FeatureFunctionTransition(FeatureFunctionStateTo):
    cdef long state_from;
    def __init__(self, long state_from, long state_to):
        super().__init__(state_to);
        self.state_from = state_from;
        
    cdef double calcStateTo(self, long state_from, double[:, :] observations, long t):
        if state_from == self.state_from:
            return 1.0;
        else:
            return 0.0;
        
cdef class FeatureFunctionObservation(FeatureFunctionStateToIndexedCurrentObservation):
    cdef long observation
    def __init__(self, long state_to, long observationIndex, long observation):
        super().__init__(state_to,observationIndex);
        self.observation = observation;
        
    cdef double calcStateToIndexedCurrentObservation(self, long state_from, double current_observation):
        if current_observation == self.observation:
            return 1.0;
        else:
            return 0.0;
        
cdef class FeatureFunctionObservationTransition(FeatureFunctionObservation):
    cdef long state_from
    def __init__(self, long state_to, long state_from, long observationIndex, long observation):
        super().__init__(state_to,observationIndex,observation);
        self.state_from = state_from;
        
    cdef double calcStateToIndexedCurrentObservation(self, long state_from, double current_observation):
        if current_observation == self.observation and state_from == self.state_from:
            return 1.0;
        else:
            return 0.0;
