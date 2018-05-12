import os
import sys
import numpy as np
from prondict import prondict

import lab3_tools as tools3
import lab3_proto as proto3


sys.path.append('../lab1/')
import proto as proto1

sys.path.append('../lab2/')
import proto2 as proto2
import tools2 as tools2

ROOT = ''
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

    # Splits the train directory into two lists of links
    # One contains the links to the training data
    # and the other to the validation data
    def lists_of_paths_to_split_on(self, data):
        data_len = len(data)

        train_len = int(np.floor(.9*data_len))
        valid_len = data_len-train_len

        men_in_train = int(np.floor(train_len/2))
        women_in_train = train_len-men_in_train

        men_in_valid = int(np.floor(valid_len/2))
        women_in_valid = valid_len-men_in_valid

        training, validation = [], []

        train_file = 'tidigits/disc_4.1.1/tidigits/train'

        dirs = os.listdir(os.path.join(ROOT, train_file, 'man'))
        # For each man
        for dir in dirs:
            utterances = os.listdir(os.path.join(ROOT, train_file, 'man', dir))
            # For each utterance of the man
            for utterance in utterances:
                filename = os.path.join(ROOT, train_file, 'man', dir, utterance)

                if len(utterances) + len(training) <= men_in_train:
                    training.append(filename)
                else:
                    validation.append(filename)

        dirs = os.listdir(os.path.join(ROOT, train_file, 'woman'))
        # For each woman
        for dir in dirs:
            utterances = os.listdir(os.path.join(ROOT, train_file, 'woman', dir))

            # For each utterance of the woman
            for utterance in utterances:
                filename = os.path.join(ROOT, train_file, 'woman', dir, utterance)

                if (len(utterances) + len(training)) - men_in_train <= women_in_train:
                    training.append(filename)
                else:
                    validation.append(filename)

        return training, validation

    def extract_features(self, path, test=False):

        data = []
        for root, dirs, files in os.walk(os.path.join(
                                        ROOT, path)):
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = tools3.loadAudio(filename)

                    lmfcc = proto1.mfcc(samples)
                    mspec = proto1.mspec_only(samples)
                    targets = self._make_targets(filename, lmfcc)

                    if ('man' in filename):
                        gender = 'man'
                    else:
                        gender = 'woman'

                    # extract speakerID from root folder
                    speakerID = root[-2:]

                    data.append({
                        'filename': filename,
                        'gender': gender,
                        'speakerID': speakerID,
                        'lmfcc': lmfcc,
                        'mspec': mspec,
                        'targets': targets
                        })
        if test:
            np.savez(os.path.join(DATA, 'testdata.npz'), testdata=data)
        else:
            np.savez(os.path.join(DATA, 'traindata.npz'), traindata=data)

        return data

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
    train_path = 'tidigits/disc_4.1.1/tidigits/train'
    test_path = 'tidigits/disc_4.2.1/tidigits/test'

    # start.extract_features(train_path)
    # start.extract_features(test_path, test=True)

    training_dic_list = np.load(os.path.join(DATA, 'traindata.npz'))['traindata']

    training_list, validation_list = start.lists_of_paths_to_split_on(
                                            training_dic_list)

    final_training_dic_list = []
    final_validation_dic_list = []

    for entry in training_dic_list:
        filename_parts = entry['filename'].split('/')
        if filename_parts[0] == 'data':
            entry['filename'] = 'tidigits/'+'/'.join(filename_parts[1:])

        if entry['filename'] in training_list:
            final_training_dic_list.append(entry)

        elif entry['filename'] in validation_list:
            final_validation_dic_list.append(entry)

        else:
            raise ValueError('Uhoh')

    np.savez(os.path.join(DATA, 'train_split.npz'),
             traindata=final_training_dic_list)

    np.savez(os.path.join(DATA, 'validation_split.npz'),
             validationdata=final_validation_dic_list)
