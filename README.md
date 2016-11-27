# MaryCrf
MaryCrf is an easy to use library for linear-chain Conditional Random Fields
with freely definable feature functions. The math and algorithms will be derived
and explained in the first part. The second part shows how to use and adapt MaryCrf for your task.

Full Documentatio is [here](https://github.com/napster2202/MaryCrf/blob/master/docs/MaryCrf.pdf)

## Adder-Example
In this example we will illustrate the advantages of the more flexible feature
functions by comparing a CRF restricted to HMM-like feature functions
against a CRF with unrestricted feature functions. We choose the example of the
adder which was used to illustrate the theoretical advantages in a previous
section. Now we train both CRFs on multiple training sequences and evaluate the
results on a test sequence.

## Setup
```bash
python setup.py build_ext --inplace
```
## Coding
Import the package:
```python
from marycrf import *
```

Define feature functions of the HMM like CRF:
```python
hmmobservationFunctions = 
   [FeatureFunctionObservation(x[1],0,x[0]-1) for x in np.ndindex((3,4))] 
hmmtransitionFunctions =
   [FeatureFunctionTransition(x[1], x[0]) for x in np.ndindex((4,4))]
hmmfeaturefunctions = 
   np.array(hmmobservationFunctions + hmmtransitionFunctions)
   
hmmcrf = CyCRF(4,hmmfeaturefunctions)

```

Define feature functions of the CRF:
```python
crffeaturefunctions = 
   [FeatureFunctionObservationTransition
      (x[2], x[1],0,x[0]-1) for x in np.ndindex((3,4,4))]
      
crf = CyCRF(4,crffeaturefunctions)

```

Train both CRFs:
```python
hmmcrf.train_BFGS(train_y,train_x)
crf.train_BFGS(train_y,train_x)
```

We use the following test sequence:
```python
test_x =
    array([[-1, -1,  1,  1,  1, -1,  1,  0,  0, -1,  0, -1,  0,  0, -1, -1, -1,
        -1,  1,  0,  0,  1,  1,  1,  0, -1, -1,  1,  1,  0, -1,  1, -1, -1,
        1, -1,  0,  1,  1,  1, -1,  1,  0, -1,  1,  1,  0, -1,  1, -1]])
 
test_y =
    array([3, 3, 3, 2, 1, 0, 1, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 2, 2,
        2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 1, 0, 1])
```

Apply the Viterbi algorithm:
```python
viterbi_hmmcrf = hmmcrf.viterbi(test_x)
viterbi_crf = crf.viterbi(test_x)
```

## Results

### The real test state sequence
![The real test state sequence](https://github.com/napster2202/MaryCrf/blob/master/docs/real_1.png "The real test state sequence")

### Trace found by an HMM simulated by an CRF
![Trace found by an HMM simulated by an CRF](https://github.com/napster2202/MaryCrf/blob/master/docs/hmm_1.png "Trace found by an HMM simulated by an CRF")

### Trace found by an CRF with observation indicated transitions
![Trace found by an CRF with observation indicated transitions](https://github.com/napster2202/MaryCrf/blob/master/docs/crf_1.png "Trace found by an CRF with observation indicated transitions")
