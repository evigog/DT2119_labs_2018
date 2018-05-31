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
        p = pronDict[word] + ['sp']
        phoneTrans.append(p)

    # flatten list
    flat_list = [item for sublist in phoneTrans for item in sublist]

    # no pause model after the last word
    flat_list = flat_list[:-1]

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

    single_num_states = hmmmodels[namelist[0]]['startprob'].shape[0]
    num_pause_models = len([ut for ut in namelist if ut == 'sp'])
    num_phonemes = len(namelist) - num_pause_models
    final_num_states = num_phonemes * (single_num_states-1) + 1
    final_startprob = np.zeros(final_num_states)
    final_startprob[0] = 1

    final_transmat = np.zeros((final_num_states, final_num_states))

    pause_model = hmmmodels['sp']

    #all phonemes except sp
    clear_list = list(filter(lambda a: a != 'sp', namelist))

    before_sp = []
    for i in range(len(namelist)-1):
        if (namelist[i+1] == 'sp'):
            before_sp.append(namelist[i])

    for i in range(0, len( clear_list)):  #iterate list of phonemes
        start = i * (single_num_states-1)
        end = start + single_num_states

        #create mask for zero elements
        transmat_zero = np.array([[False, False, True, True],
                                  [True, False, False, True],
                                  [True, True, False, False],
                                  [True, True, True, True]])

        final_transmat[start:end, start:end] = hmmmodels[clear_list[i]]['transmat']
        final_transmat[start:end, start:end][transmat_zero] = 0

        if (i == 0):
            final_means = np.vstack((hmmmodels[clear_list[0]]['means']))
            final_covars = np.vstack((hmmmodels[clear_list[0]]['covars']))
        else:
            final_means = np.vstack((final_means, hmmmodels[clear_list[i]]['means']))
            final_covars = np.vstack((final_covars, hmmmodels[clear_list[i]]['covars']))

        if(i in before_sp):  #end of word is reached
            #append pause model
            final_transmat[end-2, end-1] = hmmmodels[ clear_list[i]]['transmat'][2, 3] * pause_model['startprob'][0]
            final_transmat[end-2, end] = hmmmodels[ clear_list[i]]['transmat'][2, 3] * pause_model['startprob'][1]
            final_transmat[end-1, end-1] = pause_model['transmat'][0, 0]
            final_transmat[end-1, end] = pause_model['transmat'][0, 1]

            final_means = np.vstack((final_means, hmmmodels[clear_list[i]]['means']))
            final_covars = np.vstack((final_covars, hmmmodels[clear_list[i]]['covars']))


    #convert to log space
    final_startprob = np.log(final_startprob)
    final_transmat = np.log(final_transmat[:-1, :-1])

    combinedhmm = {'name':namelist, 'means':final_means, 'startprob':final_startprob, 'covars':final_covars, 'transmat':final_transmat}

    return combinedhmm



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
