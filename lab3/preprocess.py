import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def dynamic_features(traindata):
    dynam_features_list = []

    for ut in traindata:  # for each utterance
        timesteps = len(ut['targets'])
        stacked_mfcc = np.zeros((timesteps, 7, 13))

        for t in range(timesteps - 3):

            indx = np.zeros(7, dtype=int)
            for k, i in enumerate(range(-3, 4)):
                if (t + i < 0 or t + i > timesteps):
                    indx[k] = -(t + i)  # mirrored to zero
                else:
                    indx[k] = t + i

            stacked_mfcc[t, :, :] = ut['lmfcc'][indx, :]

        dynam_features_list.append(stacked_mfcc)

    return dynam_features_list

#perform standartisation of input dataset
#mean std are standartisation parameters
def feature_standartisation( set, training, mean, std):

    if (training):
        scaler = StandardScaler()
        standarized_set = scaler.fit(set)

        return (standarized_set, standarized_set.with_mean, standarized_set.with_std)  #return paremeters to use for standartisation
                                                                                       # of validation and test
    else: #validation or test set
        scaler = StandardScaler(with_mean=mean, with_std=std)
        standarized_set = scaler.fit(set)

        return standarized_set

#create training and validation set
#def split(traindata):

#input is training, validation or test set
def flatten_set(set):

    lmfcc = set[0]['lmfcc']
    mspec = set[0]['mspec']
    target = set[0]['target']

    for i in range(1, len(set)):

        lmfcc = np.vstack(lmfcc, set[i]['lmfcc'])
        mspec = np.vstack(mspec, set[i]['mspec'])
        target = np.vstack(target, set[i]['target'])

    return {'lmfcc':lmfcc.astype('float32'), 'mspec':mspec.astype('float32'), 'target':target.astype('float32')}


def load_data():
    traindata = np.load('data/traindata.npz')['traindata']
    testdata = np.load('data/testdata.npz')['testdata']

    #train_set, val_set = split(traindata)  #todo

    #todo: standarize lmfcc and append to final list
    dynamic_lmfcc_list = dynamic_features(traindata)  ###where to use it??

    train_flat = flatten_set(train_set)
    val_flat = flatten_set(val_set)
    test_flat = flatten_set(testdata)

    #features standartisation - use same parameters with training
    std_lmfcc_train = feature_standartisation(train_flat['lmfcc'], True, 0, 0)
    mean = std_lmfcc_train[1]
    std = std_lmfcc_train[2]
    std_lmfcc_val = feature_standartisation(val_flat['lmfcc'], True, mean, std)
    std_lmfcc_test = feature_standartisation(test_flat['lmfcc'], True, mean, std)


    std_mspec_train = feature_standartisation(train_flat['mspec'], True, 0, 0)
    mean = std_mspec_train[1]
    std = std_mspec_train[2]
    std_mspec_val = feature_standartisation(val_flat['mspec'], True, mean, std)
    std_mspec_test = feature_standartisation(test_flat['mspec'], True, mean, std)

    train = {'lmfcc':std_lmfcc_train, 'mspec':std_mspec_train, 'target':train_flat['target']}
    val = {'lmfcc': std_lmfcc_val, 'mspec': std_mspec_val, 'target': val_flat['target']}
    test = {'lmfcc': std_lmfcc_test, 'mspec': std_mspec_test, 'target': train_flat['target']}


    return (train, val, test)

load_data()









