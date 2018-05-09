import os
import numpy as np
from lab3.prondict import prondict

import lab3.lab3_tools as tools3
import lab3.lab3_proto as proto3
import lab1.proto as proto1
import lab2.proto2 as proto2
import lab2.tools2 as tools2

ROOT = ''
LAB2_ROOT = os.path.join(ROOT, '..', 'lab2')
DATA = os.path.join(ROOT, 'data')


class Main:

    def __init__(self):
        lab2_models = os.path.join('lab2_models.npz')  #use updated lab2 models
        self.phoneHMMs = np.load(lab2_models)['phoneHMMs'].item()
        self.phones = sorted(self.phoneHMMs.keys())

        self.nstates = {phone: self.phoneHMMs[phone]['means'].shape[0]
                        for phone in self.phones}

        self.stateList = [ph + '_' + str(id) for ph in self.phones
                          for id in range(self.nstates[ph])]


    def extract_features(self, path, train):

        data = []
        for root, dirs, files in os.walk(os.path.join(
                                        ROOT, path)):  #tigits
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = tools3.loadAudio(filename)

                    # ...your code for feature extraction and forced alignment
                    lmfcc = proto1.mfcc(samples)
                    mspec = proto1.mspec_only(samples)
                    targets = self._make_targets(filename, lmfcc)

                    if ('man' in filename):
                        gender = 'man'
                    else:
                        gender = 'woman'

                    speakerID = root[-2:] #extract speakerID from root folder

                    # The targets we are calculating are based on the
                    # lmfcc features. Is this weird? Or is it fine,
                    # because we are looking to predict HMM states, and the
                    # HMMs are using lmfccs inherently? It's prolly fine
                    data.append({
                        'filename': filename,
                        'gender' : gender,
                        'speakerID' : speakerID,
                        'lmfcc': lmfcc,
                        'mspec': mspec,
                        'targets': targets
                        })
        if (train):
            np.savez(os.path.join(DATA, 'traindata.npz'), traindata=data)
        else:
            np.savez(os.path.join(DATA, 'testdata.npz'), testdata=data)


    def _make_targets(self, filename, lmfcc):
        wordTrans = list(tools3.path2info(filename)[2])
        phoneTrans = proto3.words2phones(wordTrans, prondict, addShortPause=True)

        utteranceHMM = proto3.concatHMMs(self.phoneHMMs, phoneTrans)

        stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                      for stateid in range(self.nstates[phone])]

        _, viterbiPath = self._viterbi(utteranceHMM, lmfcc)

        stateIndexes = []
        for state in viterbiPath:
            usid = stateTrans[int(state)]
            stateIndexes.append(self.stateList.index(usid))

        return stateIndexes

    def _viterbi(self, utteraneHMM, lmfcc):
        means = utteraneHMM['means']
        covars = utteraneHMM['covars']
        transmat = utteraneHMM['transmat']
        startprob = utteraneHMM['startprob']

        log_emlik = tools2.log_multivariate_normal_density_diag(lmfcc,
                                                                means,
                                                                covars)

        viterbi_out = proto2.viterbi(log_emlik, startprob, transmat)

        return viterbi_out['loglik'], viterbi_out['path']




if __name__ == '__main__':

    start = Main()
    train_path = 'data/disc_4.1.1/tidigits/train'
    test_path = 'data/disc_4.2.1/tidigits/test'

    #start.extract_features(train_path)
    start.extract_features(test_path, False)
