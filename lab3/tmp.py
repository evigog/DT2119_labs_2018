from prondict import prondict
import lab3_proto as proto3
import lab3_tools as tools3
import sys
sys.path.append('../lab2/')
import proto2 as proto2
import tools2 as tools2

sys.path.append('../lab1/')
import proto as proto1

import os
import numpy as np


def _viterbi(utteraneHMM, lmfcc):
    means = utteraneHMM['means']
    covars = utteraneHMM['covars']
    transmat = utteraneHMM['transmat']
    startprob = utteraneHMM['startprob']

    log_emlik = tools2.log_multivariate_normal_density_diag(lmfcc,
                                                            means,
                                                            covars)

    viterbi_out = proto2.viterbi(log_emlik, startprob, transmat)

    return viterbi_out['loglik'], viterbi_out['path']

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
    newHMM = {}
    newHMM["name"] = ""

    M = 0
    n = len(namelist)

    for i in range(n):
        newHMM["name"] += hmmmodels[namelist[i]]["name"] + ' '
        M += hmmmodels[namelist[i]]["means"].shape[0]

    newHMM["startprob"] = np.zeros(M+1)
    newHMM["transmat"] = np.zeros((M+1, M+1))

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


# ###############################################
example = np.load('lab3_example.npz')['example'][()]

phoneHMMs = np.load(os.path.join('', 'phoneHMMs.npy'))[()]
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0]
           for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = tools3.loadAudio(filename)
lmfcc = proto1.mfcc(samples)

print('Example lmfcc and computed are same: {}\n'.format(np.all(example['lmfcc'] == lmfcc)))

wordTrans = list(tools3.path2info(filename)[2])
print('Example wordTrans and computed are same: {}\n'.format(np.all(example['wordTrans'] == wordTrans)))

phoneTrans = proto3.words2phones(wordTrans, prondict)
print('Example phoneTrans and computed are same: {}\n'.format(np.all(example['phoneTrans'] == phoneTrans)))
# print(phoneTrans)
# print(example['phoneTrans'])

# utteranceHMM = proto3.concatHMMs(phoneHMMs, phoneTrans)
utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
print('For utternaceHMM')
for key in utteranceHMM.keys():
    print('\t', key)
    if key not in example['utteranceHMM']:
        print('\tNot in example utteranceHMM\n')
        continue
    print('\tExample {} and computed {} are the same: {}\n'.format(
                    key, key,
                    np.all(utteranceHMM[key] == example['utteranceHMM'][key])))

stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
              for stateid in range(nstates[phone])]
print('Example stateTrans and computed are same: {}\n'.format(np.all(example['stateTrans'] == stateTrans)))

viterbiLogLik, viterbiPath = _viterbi(utteranceHMM, lmfcc)

print('Example viterbiLogLik and computed are same: {}\n'.format(np.all(example['viterbiLoglik'] == viterbiLogLik)))
print('Example viterbiPath and computed are same: {}\n'.format(np.all(example['viterbiPath'] == viterbiPath)))

stateIndexes = []
for state in viterbiPath:
    usid = stateTrans[int(state)]
    stateIndexes.append(stateList.index(usid))

print('Example viterbiStateTrans and computed are same: {}\n'.format(np.all(example['viterbiStateTrans'] == stateIndexes)))