import os
from utilities import Constants
from utilities import Logging
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping

from keras.utils.np_utils import to_categorical

co = Constants()
logger = Logging()


def get_data(set_name):
    if set_name not in ['train', 'test', 'validation']:
        raise ValueError(("Pls define set name as: \'train\', "
                         "\'test\', \'validation\' only"))

    filename = '{}_preprocessed.npz'.format(set_name)

    archive = np.load(os.path.join(co.DATA_ROOT, filename))

    return archive


def define_network(input_shape):

    model = Sequential()

    model.add(Dense(256, input_dim=input_shape[1]))
    model.add(Activation('relu'))

    for layer in range(co.HIDDEN_LAYERS-1):
        model.add(Dense(256))
        model.add(Activation('relu'))

    model.add(Dense(co.NUM_CLASSES))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=co.LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=categorical_crossentropy,
                  metrics=['accuracy']
                  )

    return model


def train_network(model, train_X, train_Y, valid_X, valid_Y):
    early_stopping = EarlyStopping(monitor='val_acc',
                                   min_delta=1e-2, patience=2)

    hist = model.fit(train_X, train_Y,
                     validation_data=(valid_X, valid_Y),
                     epochs=co.EPOCHS, batch_size=co.BATCH_SIZE,
                     callbacks=[early_stopping])

    return model, hist


def store_model(model):

    #create one path for each new model
    if not os.path.exists(co.MODELS_ROOT):
        os.mkdir(co.MODELS_ROOT)

    already_stored = os.listdir(co.MODELS_ROOT)
    if len(already_stored) != 0:
        # Split model name, get last part #.h5, extract only #
        ids = [int(model_name).split('_')[-1][:-3]
               for model_name in already_stored]

        model_name = '{}_model_{}.h5'.format(co.INPUT_KIND, np.amax(ids)+1)
    else:
        model_name = '{}_model_{}.h5'.format(co.INPUT_KIND, '0')

    model.save(os.path.join(co.MODELS_ROOT, model_name))
    return os.path.join(co.MODELS_ROOT, model_name)

def training_pipeline(train_feature):  #train_feature: lmfcc, mspec, dynamic_lmfcc, dynamic_mspec

    co.INPUT_KIND = train_feature
    training_dictionary = get_data('train')
    validation_dictionary = get_data('validation')

    training_lmfcc = training_dictionary[co.INPUT_KIND]
    training_targets = training_dictionary['targets']
    training_categorical_targets = to_categorical(training_targets)

    validation_lmfcc = validation_dictionary[co.INPUT_KIND]
    validation_targets = validation_dictionary['targets']
    validation_categorical_targets = to_categorical(validation_targets)

    input_shape = np.shape(training_lmfcc)

    model = define_network(input_shape)
    model, hist = train_network(model,
                                training_lmfcc,
                                training_categorical_targets,
                                validation_lmfcc,
                                validation_categorical_targets)

    model_location = store_model(model)

    test_dictionary = get_data('test')
    test_lmfcc = test_dictionary[co.INPUT_KIND]
    test_targets = test_dictionary['targets']
    test_categorical_targets = to_categorical(test_targets)

    evaluation = model.evaluate(test_lmfcc, test_categorical_targets)
    print('Evaluation: ', evaluation)

    entry = {
        'model_location': model_location,
        'input': co.INPUT_KIND,
        'val_loss': hist.history['val_loss'],
        'loss': hist.history['loss'],
        'val_acc': hist.history['val_acc'],
        'acc': hist.history['acc'],
        'test_loss': evaluation[0],
        'test_acc': evaluation[1],
        'epochs_trained': len(hist.history['loss']),
        'training_params': co.net_params_to_dictionary(),
    }

    logger.store_log_entry(entry)

if __name__ == '__main__':
#
    feature_list = ['lmfcc', 'mspec', 'dynamic_lmfcc', 'dynamic_mspec']
    for feature in feature_list:
        training_pipeline(feature)
#
    #read log file and do plotting
    log_feed = logger.read_log()

    for model in log_feed:

        #plot training and validation loss across epochs
        epochs_range = range(0, model['epochs_trained'])
        plt.plot(epochs_range, model['loss'], label='Training loss')
        plt.plot(epochs_range, model['val_loss'], label='Validation loss')
        plt.title('Training and Validation loss for model %s' %model['input'])
        plt.legend()
        plt.savefig('models/loss_%s'%model['input'] )
        plt.show()


        # plot training and validation accuracy across epochs
        epochs_range = range(0, model['epochs_trained'])
        plt.plot(epochs_range, model['loss'], label='Training accuracy')
        plt.plot(epochs_range, model['val_loss'], label='Validation accuracy')
        plt.title('Training and Validation accuracy for model %s' % model['input'])
        plt.legend()
        plt.savefig('models/accuracy_%s' % model['input'])
        plt.show()





