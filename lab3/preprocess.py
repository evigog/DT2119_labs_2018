import os
from lab3.utilities import Constants
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self):
        self.co = Constants()

    # create dynamic features of input matrix
    def _dynamic_features(self, stack):
        dynam_features_list = []

        timesteps = stack.shape[0]
        for t in range(timesteps):

            indx = np.zeros(7, dtype=int)

            for k, i in enumerate(range(-3, 4)):
                if (t + i < 0 ):
                    # mirrored to zero
                    indx[k] = -(t + i)
                elif (t+i > timesteps-1):
                    indx[k] = t - i
                else:
                    indx[k] = t + i

            if (t > 85 ):
                a = [t-3, t-2, t-1, t, t+1, t+2, t+3]
                b = indx
                z = a

            dynam_features_list.append(stack[indx, :])

        return dynam_features_list

    # set is a list of dictionaries. Each dictionary contains a sample
    def _create_dynamic_features(self, set):

        for sample in set:
            # lmfcc_f is TxD, timestepsxfeatures
            lmfcc_f = sample['lmfcc']
            mspec_f = sample['mspec']

            dynamic_lmfcc = self._get_sample_dynamic_vectors(lmfcc_f)
            dynamic_mspec = self._get_sample_dynamic_vectors(mspec_f)

            # print('d lmfcc: ', np.shape(dynamic_lmfcc))
            # print('d mspec: ', np.shape(dynamic_mspec))

            sample['dynamic_lmfcc'] = dynamic_lmfcc
            sample['dynamic_mspec'] = dynamic_mspec

        return set

    # That is *almost* correct at the boundaries.
    def _get_sample_dynamic_vectors(self, sample_features):
            sample_dynamic_vectors = []

            for t in range(np.shape(sample_features)[0]):
                current = sample_features[t, :]
                # We want the previous and next 3 features around
                # the current timestep
                if t < 3:
                    next = [el for vector in [sample_features[i, :]
                            for i in range(t+1, t+4)] for el in vector]
                    if t == 0:
                        prev_slice = sample_features[t+1:t+4, :][::-1]
                    else:
                        prev_slice = sample_features[t-1: t+2, :][::-1]

                    previous = [el for vector in prev_slice for el in vector]
                elif t >= np.shape(sample_features)[1]-3:
                    next_slice = sample_features[-3:, :]
                    previous = [el for vector in [sample_features[i, :]
                                for i in range(t-3, t)] for el in vector]

                    next = [el for vector in next_slice for el in vector]

                else:
                    prev_slice = sample_features[t-3:t, :]
                    next_slice = sample_features[t+1:t+4, :]

                    previous = [el for vector in prev_slice for el in vector]
                    next = [el for vector in next_slice for el in vector]

                # print('previous: ', np.shape(previous))
                # print('current: ', np.shape(current))
                # print('next: ', np.shape(next))

                dynamic_feature_vector = np.hstack((previous, current, next))

                sample_dynamic_vectors.append(dynamic_feature_vector)

            return sample_dynamic_vectors

    def _flatten(self, set):
        lmfcc_stack, mspec_stack, dynamic_lmfcc, dynamic_mspec, targets_stack = [], [], [], [], []

        for entry in set:
            for lmfcc in entry['lmfcc']:
                lmfcc_stack.append(lmfcc)
            for mspec in entry['mspec']:
                mspec_stack.append(mspec)
            for d_lmfcc in entry['dynamic_lmfcc']:
                dynamic_lmfcc.append(d_lmfcc)
            for d_mspec in entry['dynamic_mspec']:
                dynamic_mspec.append(d_mspec)
            for target in entry['targets']:
                targets_stack.append(target)

        lmfcc_stack = np.asarray(lmfcc_stack)
        mspec_stack = np.asarray(mspec_stack)
        dynamic_lmfcc = np.asarray(dynamic_lmfcc)
        dynamic_mspec = np.asarray(dynamic_mspec)
        targets_stack = np.asarray(targets_stack)

        return self._float_convert(lmfcc_stack), \
            self._float_convert(mspec_stack),    \
            self._float_convert(dynamic_lmfcc),  \
            self._float_convert(dynamic_mspec),  \
            self._float_convert(targets_stack)

    def _float_convert(self, arr):
        if not type(arr) == np.ndarray:
            raise ValueError('Pls only np array, k?')

        return arr.astype('float32')

    def _standardize_training(self, lmfcc_stack, mspec_stack,
                              dynamic_lmfcc_stack, dynamic_mspec_stack):

        scaler = StandardScaler()

        scaled_lmfcc = scaler.fit_transform(lmfcc_stack)
        lmfcc_mean = scaler.mean_
        lmfcc_std = scaler.var_

        scaled_mspec = scaler.fit_transform(mspec_stack)
        mspec_mean = scaler.mean_
        mspec_std = scaler.var_

        scaled_dynamic_lmfcc = scaler.fit_transform(dynamic_lmfcc_stack)
        dynamic_lmfcc_mean = scaler.mean_
        dynamic_lmfcc_std = scaler.var_

        scaled_dynamic_mspec = scaler.fit_transform(dynamic_mspec_stack)
        dynamic_mspec_mean = scaler.mean_
        dynamic_mspec_std = scaler.var_

        return (scaled_lmfcc, lmfcc_mean, lmfcc_std), \
               (scaled_mspec, mspec_mean, mspec_std), \
               (scaled_dynamic_lmfcc, dynamic_lmfcc_mean, dynamic_lmfcc_std), \
               (scaled_dynamic_mspec, dynamic_mspec_mean, dynamic_mspec_std)

    def _standardize_rest(self, lmfcc_stack, mspec_stack,
                          dynamic_lmfcc_stack, dynamic_mspec_stack):

            scaled_lmfcc = self._do_math(lmfcc_stack, self.train_lmfcc_mean,
                                         self.train_lmfcc_std)

            scaled_mspec = self._do_math(mspec_stack, self.train_mspec_mean,
                                         self.train_mspec_std)

            scaled_dynamic_lmfcc = self._do_math(dynamic_lmfcc_stack,
                                                 self.train_dynamic_lmfcc_mean,
                                                 self.train_dynamic_lmfcc_std)

            scaled_dynamic_mspec = self._do_math(dynamic_mspec_stack,
                                                 self.train_dynamic_mspec_mean,
                                                 self.train_dynamic_mspec_std)

            return scaled_lmfcc, scaled_mspec, scaled_dynamic_lmfcc, scaled_dynamic_mspec

    def _do_math(self, set, mean, std):
        return (set - mean)/std

    def process(self, set_name):
        print('Loading unprocessed splits...')
        if set_name == 'train':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'train_split.npz'))['traindata']
        elif set_name == 'validation':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'validation_split.npz'))['validationdata']
        elif set_name == 'test':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'testdata.npz'))['testdata']

        print('Creating dynamic features...')
        set = self._create_dynamic_features(set)

        print('Flattening...')
        set_lmfcc, set_mspec, set_dynamic_lmfcc, \
            set_dynamic_mspec, set_targets = self._flatten(set)

        print('dynamic lmfcc: ', np.shape(set_dynamic_lmfcc))
        print('dynamic_mspec: ', np.shape(set_dynamic_mspec))

        print('Scaling...')
        if set_name == 'train':
            set_lmfcc_scaling_t, set_mspec_scaling_t, \
                set_dynamic_lmfcc_scaling_t, set_dynamic_mspec_scaling_t = \
                self._standardize_training(set_lmfcc, set_mspec,
                                           set_dynamic_lmfcc,
                                           set_dynamic_mspec)

            set_lmfcc = set_lmfcc_scaling_t[0]
            set_mspec = set_mspec_scaling_t[0]

            set_dynamic_lmfcc = set_dynamic_lmfcc_scaling_t[0]
            set_dynamic_mspec = set_dynamic_mspec_scaling_t[0]

            self.train_lmfcc_mean = set_lmfcc_scaling_t[1]
            self.train_lmfcc_std = set_lmfcc_scaling_t[2]

            self.train_mspec_mean = set_mspec_scaling_t[1]
            self.train_mspec_std = set_mspec_scaling_t[2]

            self.train_dynamic_lmfcc_mean = set_dynamic_lmfcc_scaling_t[1]
            self.train_dynamic_lmfcc_std = set_dynamic_lmfcc_scaling_t[2]

            self.train_dynamic_mspec_mean = set_dynamic_mspec_scaling_t[1]
            self.train_dynamic_mspec_std = set_dynamic_mspec_scaling_t[2]

        else:
            set_lmfcc, set_mspec, set_dynamic_lmfcc, set_dynamic_mspec = \
                                self._standardize_rest(
                                    set_lmfcc, set_mspec,
                                    set_dynamic_lmfcc,
                                    set_dynamic_mspec)

        self._store(set_lmfcc, set_mspec, set_dynamic_lmfcc, set_dynamic_mspec,
                    set_targets, set_name)
        print()

    def _store(self, set_lmfcc, set_mspec, set_dynamic_lmfcc,
               set_dynamic_mspec, set_targets, set_name):

        filename = '{}_preprocessed.npz'.format(set_name)

        np.savez(os.path.join(self.co.DATA_ROOT, filename),
                 lmfcc=set_lmfcc, mspec=set_mspec,
                 dynamic_lmfcc=set_dynamic_lmfcc,
                 dynamic_mspec=set_dynamic_mspec,
                 targets=set_targets)


if __name__ == '__main__':
    preprocessor = Preprocessor()

    preprocessor.process('train')
    preprocessor.process('validation')
    preprocessor.process('test')
