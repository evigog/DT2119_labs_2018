import numpy as np
from lab3_tools import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phoneTrans = []
    for word in wordList:
        # p = pronDict[word] + ['sp']
        p = pronDict[word]
        phoneTrans.append(p)

    # flatten list
    flat_list = [item for sublist in phoneTrans for item in sublist]

    # no pause model after the last word
    # flat_list = flat_list[:-1]

    # add silence in beginnind and end
    flat_list = ['sil'] + flat_list
    flat_list.append(('sil'))

    return flat_list

def concatHMMs(hmmmodels, namelist):  #(phoneHMMs, phoneTrans)
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """

    newHMM = {}
    newHMM["name"] = ""

    M = 0
    n = len(namelist)

    for i in range(n):
        newHMM["name"] += hmmmodels[namelist[i]]["name"] + ' '
        M += hmmmodels[namelist[i]]["means"].shape[0]

    newHMM["startprob"] = np.zeros(M + 1)
    newHMM["transmat"] = np.zeros((M + 1, M + 1))

    start_i = 0
    for i in range(n):
        if i == 0:
            newHMM["startprob"][:len(hmmmodels[namelist[i]]["startprob"])] = hmmmodels[namelist[i]]["startprob"]
            newHMM["means"] = hmmmodels[namelist[i]]["means"]
            newHMM["covars"] = hmmmodels[namelist[i]]["covars"]
        else:
            newHMM["means"] = np.vstack((newHMM["means"], hmmmodels[namelist[i]]["means"]))
            newHMM["covars"] = np.vstack((newHMM["covars"], hmmmodels[namelist[i]]["covars"]))

        z = hmmmodels[namelist[i]]["transmat"].shape[0]
        mat = hmmmodels[namelist[i]]["transmat"]
        for j in range(z):
            for k in range(z):
                newHMM["transmat"][j + start_i][k + start_i] = mat[j, k]
        start_i += (z - 1)

    return newHMM


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """



def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
