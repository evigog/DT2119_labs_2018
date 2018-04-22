import numpy as np
from tools2 import *

import matplotlib.pyplot as plt

def concatHMMs(hmmmodels, namelist):
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

    M, D = np.shape(hmmmodels['ow']['covars'])
    num_phonems = len(namelist)
    # num_phonemes*M+1 is the len of the new transition matrix

    new_transition_matrix = np.zeros(((3*M)+1, (3*M)+1))
    new_means = np.zeros((len(hmmmodels)*M, D))
    new_covars = np.zeros(np.shape(new_means))
    new_start_prob = np.zeros((3*M))

    combinedhmm = {}
    for i, phoneme in enumerate(namelist):
        hmm = hmmmodels[phoneme]

        dim = np.shape(hmm['transmat'])[0]

        if i == 0:
            start = i*dim
            end = (i*dim)+dim
            new_transition_matrix[start:end, start:end] = hmm['transmat']
        elif i == 1:
            start = (i*dim)-1
            end = (i*dim)+dim-1
            new_transition_matrix[start:end, start:end] = hmm['transmat']
        elif i == 2:
            start = (i*dim)-2
            end = (i*dim)+dim-2
            new_transition_matrix[start:, start:] = hmm['transmat']

        new_means[i*M:i*M+3] = hmm['means']
        new_covars[i*M:i*M+3] = hmm['covars']
        new_start_prob[i*M:i*M+3] = hmm['startprob'][:-1]

    combinedhmm['name'] = namelist[-2]
    combinedhmm['startprob'] = new_start_prob
    combinedhmm['transmat'] = new_transition_matrix
    combinedhmm['means'] = new_means
    combinedhmm['covars'] = new_covars

    return combinedhmm

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    # N frames, M states
    N, M = np.shape(log_emlik)

    print('shape log_emlik: ', np.shape(log_emlik))
    print('shape log_startprob: ', np.shape(log_startprob))
    print('shape log_transmat: ', np.shape(log_transmat))

    forward_prob = np.zeros((N, M))

    log_alpha_0 = np.add(log_startprob, log_emlik[0, :])
    forward_prob[0, :] = log_alpha_0

    for timestep in range(1, N):
        for state in range(M):
            log_alpha = logsumexp(np.add(forward_prob[timestep-1, :],
                                         log_transmat[state, :]
                                         )
                                  ) + log_emlik[timestep, :]

        forward_prob[timestep, :] = log_alpha

    print('shape forward_prob: ', np.shape(forward_prob))

    return forward_prob

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
