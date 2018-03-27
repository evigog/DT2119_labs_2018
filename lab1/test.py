from lab1.proto import *
import numpy as np


def example_mfcc():

    example = np.load('lab1_example.npz')['example'].item()

    #computed coef for example
    coef = mfcc(example['samples'])

    #true output
    myplot(example['frames'], 'True frames')
    myplot(example['preemph'], 'True preemphasis')
    myplot(example['windowed'], 'True windows')
    myplot(example['spec'], 'True spectrum')
    myplot(example['mspec'], 'True Mel filterbank')
    myplot(example['mfcc'], 'True Cepstrum Coefficients')
    myplot(example['lmfcc'], 'True Lifter Cepstrum Coefficients')

    return coef

def compute_mfcc():

     data = np.load('lab1_data.npz')['data']
     coeffs = mfcc(data[0]['samples'])
     for i in range(1,44):
         c = mfcc(data[i]['samples'])
         coeffs = np.append(coeffs, c, axis=0)
         a = coeffs

     return coeffs






#coef = example_mfcc()
coeffs = compute_mfcc()


