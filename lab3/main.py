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

    def split_training_data(self):
        data = self.extract_features()
        data_len = len(data)

        train_len = int(np.floor(.9*data_len))
        valid_len = data_len-train_len

        men_in_train = int(np.floor(train_len/2))
        women_in_train = train_len-men_in_train

        men_in_valid = int(np.floor(valid_len/2))
        women_in_valid = valid_len-men_in_valid

        training, validation = []

        train_file = 'tidigits/disc_4.1.1/tidigits/train'

        men_added, women_added = 0
        for root, dirs, files in os.walk(
                                    os.path.join(ROOT, train_file, 'man')):
                for dir in np.shuffe(dirs):
                    utterances = os.listdir(
                                    os.path.join(ROOT, train_file, 'man', dir))

                    for utterance in np.shuffle(utterances):
                        filename = os.path.join(root, dir, utterance)

                        if len(utterances) + len(training) <= men_in_train:
                            training.append(filename)
                        else:
                            validation.append(filename)
                
                # Stops walking into further subdirectories
                break

    # def _extract_gender_balanced_filename_lists(self, )








    def extract_features(self, test=False):

        paths = ['tidigits/disc_4.1.1/tidigits/train', 'tidigits/disc_4.2.1/tidigits/test']

        if test:
            out_filename = 'test_data.npz'
            path = paths[1]
        else:
            out_filename = 'train_data.npz'
            path = paths[0]

        data = []
        for root, dirs, files in os.walk(
                                os.path.join(ROOT, path)):
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = tools3.loadAudio(filename)

                    lmfcc = tools1.mfcc(samples)
                    mspec = tools1.mspec_only(samples)
                    targets = self._make_targets(filename, lmfcc)

                    data.append({
                        'filename': filename,
                        'lmfcc': lmfcc,
                        'mspec': mspec,
                        'targets': targets
                        })

        if test:
            np.savez(os.path.join(DATA, out_filename), testdata=data)
        else:
            np.savez(os.path.join(DATA, out_filename), traindata=data)

        return data

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
