import os
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = ''
DATA = os.path.join(ROOT, 'data')

#create dynamic features of input matrix
def dynamic_features(matrix):
    dynam_features_list = []

    timesteps = matrix.shape[0]
    for t in range(timesteps):

        indx = np.zeros(7, dtype=int)

        for k, i in enumerate(range(-3, 4)):
            if (t + i < 0 ):
                indx[k] = -(t + i)  # mirrored to zero
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


#create dynamic features of input set and update set (training, validation, testing)
def create_dynamic_features(dataset):

    for data in dataset:
        dynamic_lmfcc = dynamic_features(data['lmfcc'])
        dynamic_mspec = dynamic_features(data['mspec'])

        data['dynamic-lmfcc'] = dynamic_lmfcc
        data['dynamic-mspec'] = dynamic_mspec

    return dataset


#perform standartisation of input dataset
#mean std are standartisation parameters
def feature_standartisation( set, training, mean, std):

    if (training):
        scaler = StandardScaler()
        standarized_set = scaler.fit_transform(set)

        return (standarized_set, scaler.mean_, scaler.var_)  #return paremeters to use for standartisation
                                                                                       # of validation and test
    else: #validation or test set
        scaler = StandardScaler(with_mean=mean, with_std=std)
        standarized_set = scaler.fit_transform(set)

        return standarized_set


def create_standardized_set(training_feature, validation_feature, testing_feature):

    std_feature_train = feature_standartisation(training_feature, True, 0, 0)
    mean = std_feature_train[1]
    std = std_feature_train[2]
    #standrdize validation adn testing with the same parameters as training
    std_feature_val = feature_standartisation(validation_feature, True, mean, std)
    std_feature_test = feature_standartisation(testing_feature, True, mean, std)

    return (std_feature_train[0], std_feature_val[0], std_feature_test[0])


#create training and validation set
#def split(traindata):

#input is training, validation or test set
def flatten_set(set):

   lmfcc = set[0]['lmfcc']
   mspec = set[0]['mspec']
   target = np.array(set[0]['targets']).reshape((-1, 1))
   dynamic_lmfcc = set[0]['dynamic-lmfcc']
   dynamic_mspec = set[0]['dynamic-mspec']

   for i in range(1, len(set)):

        lmfcc = np.vstack((lmfcc, set[i]['lmfcc']))
        mspec = np.vstack((mspec, set[i]['mspec']))
        target_vec = np.array(set[i]['targets']).reshape((-1, 1))
        target = np.vstack((target,  target_vec))
        dynamic_lmfcc = np.vstack((dynamic_lmfcc, set[i]['dynamic-lmfcc']))
        dynamic_mspec = np.vstack((dynamic_mspec, set[i]['dynamic-mspec']))

   #create 2D matrices
   dynamic_lmfcc = np.reshape(dynamic_lmfcc, (-1, 91))
   dynamic_mspec = np.reshape(dynamic_mspec, (-1, 280))

   return {'lmfcc':lmfcc.astype('float32'), 'mspec':mspec.astype('float32'), 'dynamic-lmfcc':dynamic_lmfcc.astype('float32'),
           'dynamic-mspec':dynamic_mspec.astype('float32'), 'targets':target.astype('float32')}

def load_data():
    traindata = np.load('data/training_split.npz')['traindata']
    validationdata = np.load('data/validation_split.npz')['validationdata']
    testdata = np.load('data/testdata.npz')['testdata']

    #create dynamic features for training data
    traindata = create_dynamic_features(traindata)
    validationdata = create_dynamic_features(validationdata)
    testdata = create_dynamic_features(testdata)


    #flatten lmfcc, mspec, dynamic lmfcc, dynamic mspec
    train_flat = flatten_set(traindata)
    val_flat = flatten_set(validationdata)
    test_flat = flatten_set(testdata)

    #create standardized features
    std_lmfcc_train, std_lmfcc_val, std_lmfcc_test = create_standardized_set(train_flat['lmfcc'], val_flat['lmfcc'], test_flat['lmfcc'])
    std_mspec_train, std_mspec_val, std_mspec_test = create_standardized_set(train_flat['mspec'], val_flat['mspec'],test_flat['mspec'])

    std_dynam_lmfcc_train, std_dynam_lmfcc_val, std_dynam_lmfcc_test = create_standardized_set(train_flat['dynamic-lmfcc'], val_flat['dynamic-lmfcc'],
                                                                           test_flat['dynamic-lmfcc'])

    std_dynam_mspec_train, std_dynam_mspec_val, std_dynam_mspec_test = create_standardized_set(train_flat['dynamic-mspec'], val_flat['dynamic-mspec'],
                                                                           test_flat['dynamic-mspec'])


    train = {'lmfcc':std_lmfcc_train, 'mspec':std_mspec_train, 'dynamic-lmfcc':std_dynam_lmfcc_train, 'dynamic-mspec':std_dynam_mspec_train, 'target':train_flat['targets']}
    val = {'lmfcc': std_lmfcc_val, 'mspec': std_mspec_val, 'dynamic-lmfcc':std_dynam_lmfcc_val, 'dynamic-mspec':std_dynam_mspec_val, 'target': val_flat['targets']}
    test = {'lmfcc': std_lmfcc_test, 'mspec': std_mspec_test, 'dynamic-lmfcc':std_dynam_lmfcc_test, 'dynamic-mspec':std_dynam_mspec_test, 'target': test_flat['targets']}


    np.savez(os.path.join(DATA, 'train_preprocessed.npz'), traindata=train)
    np.savez(os.path.join(DATA, 'val_preprocessed.npz'), validationdata=val)
    np.savez(os.path.join(DATA, 'test_preprocessed.npz'), testdata=test)


load_data()











