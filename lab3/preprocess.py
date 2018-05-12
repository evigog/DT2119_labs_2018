import os
from utilities import Constants
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self):
        self.co = Constants()

    # create dynamic features of input matrix
    def _dynamic_features(self, matrix):
        dynam_features_list = []

        timesteps = matrix.shape[0]
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

            dynam_features_list.append(matrix[indx, :])

        return dynam_features_list

    def _flatten(self, set):
        lmfcc_stack, mspec_stack, targets_stack = [], [], []

        for entry in set:
            for lmfcc in entry['lmfcc']:
                lmfcc_stack.append(lmfcc)
            for mspec in entry['mspec']:
                mspec_stack.append(mspec)
            for target in entry['targets']:
                targets_stack.append(target)

        lmfcc_stack = np.array(lmfcc_stack)
        mspec_stack = np.array(mspec_stack)
        targets_stack = np.array(targets_stack)

        return self._float_covert(lmfcc_stack), self._float_covert(mspec_stack), self._float_covert(targets_stack)

    def _float_covert(self, arr):
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

        scaled_dynamic_lmfcc = scaler.fit(dynamic_lmfcc_stack)
        dynamic_lmfcc_mean = scaler.mean_
        dynamic_lmfcc_std = scaler.var_

        scaled_dynamic_mspec = scaler.fit(dynamic_mspec_stack)
        dynamic_mspec_mean = scaler.mean_
        dynamic_mspec_std = scaler.var_

        return (scaled_lmfcc, lmfcc_mean, lmfcc_std), \
               (scaled_mspec, mspec_mean, mspec_std), \
               (scaled_dynamic_lmfcc, dynamic_lmfcc_mean, dynamic_lmfcc_std), \
               (scaled_dynamic_mspec, dynamic_mspec_mean, dynamic_mspec_std)

    def _standardize_rest(self, lmfcc_stack, mspec_stack,
                          dynamic_lmfcc_stack, dynamic_mspec_stack):

            scaled_lmfcc = self._do_maths(lmfcc_stack, self.train_lmfcc_mean,
                                         self.train_lmfcc_std)

            scaled_mspec = self._do_maths(mspec_stack, self.train_mspec_mean,
                                         self.train_mspec_std)

            scaled_dynamic_lmfcc = self._do_maths(dynamic_lmfcc_stack,
                                                 self.train_dynamic_lmfcc_mean,
                                                 self.train_dynamic_lmfcc_std)

            scaled_dynamic_mspec = self._do_maths(dynamic_mspec_stack,
                                                 self.train_dynamic_mspec_mean,
                                                 self.train_dynamic_mspec_std)

            return scaled_lmfcc, scaled_mspec, scaled_dynamic_lmfcc, scaled_dynamic_mspec

    def _do_maths(self, set, mean, std):
        return (set - mean)/std

    def process(self, set_name):
        print('Loading unprocessed splits...')
        if set_name == 'train':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'train_split.npz'))['traindata']
        elif set_name == 'validation':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'validation_split.npz'))['validationdata']
        elif set_name == 'test':
            set = np.load(os.path.join(self.co.DATA_ROOT, 'testdata.npz'))['testdata']

        print('Flattening...')
        set_lmfcc, set_mspec, set_targets = self._flatten(set)

        print('Creating dynamic features...')
        set_dynamic_lmfcc = self._dynamic_features(set_lmfcc)
        set_dynamic_mspec = self._dynamic_features(set_mspec)

        print('dynamic lmfcc: ', np.shape(set_dynamic_lmfcc))
        print('dynamic_mspec: ', np.shape(set_dynamic_mspec))

        print('Scaling...')
        if set_name == 'train':
            set_lmfcc_scaling_t, set_mspec_scaling_t, set_dynamic_lmfcc_scaling_t, set_dynamic_mspec_scaling_t = \
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
