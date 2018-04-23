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
    num_states = log_transmat.shape[0]
    num_observations = log_emlik.shape[0]

    backward_prob = np.zeros((num_observations, num_states)) #initialize with zero

    for t in range(num_observations-2, -1, -1):
        for s in range(num_states):
            #compute sum
            sum_mat = log_transmat[s,:] + log_emlik[t+1,:] + backward_prob[t+1, :]
            backward_prob[t, s] = logsumexp(sum_mat)

    return backward_prob


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
    num_states = log_transmat.shape[0]
    num_observations = log_emlik.shape[0]   #equal to number of frames
    path_matrix = np.zeros((num_states, num_observations))
    backpointer = np.zeros((num_states, num_observations)) #best previous path for each time step

    #initialization step
    for s in range(num_states-1): #forget about last state
        path_matrix[s, 0] = log_startprob[s] + log_emlik[0,s]
    #recursion step
    for t in range(1, num_observations):
       for s in range(num_states):
           v = path_matrix[:, t - 1] + log_transmat[:, s]
           best = np.argmax(v)
           path_matrix[s,t] = path_matrix[best,t-1] + log_transmat[best, s] + log_emlik[t, s]
           backpointer[s, t] = best

    backpointer[-1, -1] = np.argmax(path_matrix[s, num_observations-1] + log_transmat[best, -1])
    viterbi_loglik = np.max(path_matrix[:, -1])
    # backtracking
    backpointer = backpointer.astype(int)
    viterbi_path = np.zeros((num_observations))
    viterbi_path[0] = 0
    viterbi_path[-1] = int(np.argmax(path_matrix[:, -1]))

    for t in range(num_observations-2, -1, -1):
       best = backpointer[int(viterbi_path[t+1]), t+1]
       viterbi_path[t]= int(best)

    return viterbi_loglik, viterbi_path


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
