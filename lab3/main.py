import os
import numpy as np
from prondict import prondict

import lab3_tools as tools3
import lab3_proto as proto3

ROOT = './'
LAB2_ROOT = os.path.join(ROOT, '..', 'lab2')
DATA = os.path.join(ROOT, 'data')


class Main:

    def __init__(self):
        lab2_models = os.path.join(LAB2_ROOT, 'lab2_models.npz')
        self.phoneHMMs = np.load(lab2_models)['phoneHMMs'].item()
        self.phones = sorted(self.phoneHMMs.keys())

        self.nstates = {phone: self.phoneHMMs[phone]['means'].shape[0]
                        for phone in self.phones}

        self.stateList = [ph + '_' + str(id) for ph in self.phones
                          for id in range(self.nstates[ph])]



    def extract_features(self):

        traindata = []
        for root, dirs, files in os.walk(os.path.join(
                                        ROOT,
                                        'tidigits/disc_4.1.1/tidigits/train')):
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = tools3.loadAudio(filename)

                    # ...your code for feature extraction and forced alignment
                    lmfcc = tools1.mfcc(samples)
                    mspec = tools1.mspec_only(samples)
                    targets = self._make_targets(filename, lmfcc)

                    # The targets we are calculating are based on the
                    # lmfcc features. Is this weird? Or is it fine,
                    # because we are looking to predict HMM states, and the
                    # HMMs are using lmfccs inherently? It's prolly fine
                    traindata.append({
                        'filename': filename,
                        'lmfcc': lmfcc,
                        'mspec': mspec,
                        'targets': targets
                        })

        np.savez(os.path.join(DATA, 'traindata.npz'), traindata=traindata)

    def _make_targets(self, filename):
        wordTrans = list(tools3.path2info(filename)[2])
        phoneTrans = proto3.words2phones(wordTrans, prondict)

        utteraneHMM = proto3.concatHMMs(self.phoneHMMs, phoneTrans,
                                        addShortPause=False)

        stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                      for stateid in range(self.nstates[phone])]

        _, viterbiPath = self._viterbi(utteraneHMM)

        stateIndexes = []
        for state in viterbiPath:
            usid = stateTrans[state]
            stateIndexes.append(self.stateList.index(usid))

    def _viterbi(self, utteraneHMM, lmfcc):
        means = utteraneHMM['means']
        covars = utteraneHMM['covars']
        transmat = utteraneHMM['transmat']
        startprob = utteraneHMM['startprob']

        log_emlik = tools2.log_multivariate_normal_density_diag(lmfcc,
                                                                means,
                                                                covars)

        startprob = utteraneHMM['startprob']
        transmat = utteraneHMM['transmat']

        viterbi_out = proto2.viterbi(log_emlik, startprob, transmat)

        return viterbi_out['loglik'], viterbi_out['path']


if __name__ == '__main__':
    start = Main()
