import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import lab3_tools as tools3

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from preprocess import Preprocessor
from utilities import Constants
from utilities import Logging


INPUT_KIND = None


def substitute_phonemes_with_generic_name(list_of_phonemes):
    subed = []
    for ph in list_of_phonemes:
        ph_name = ph.split('_')[0]
        subed.append(ph_name)

    return subed


def make_transcription(list_of_phonemes):
    transcription = []
    transcription.append(list_of_phonemes[0])
    for i in range(len(list_of_phonemes[1:])):
        if transcription[-1] != list_of_phonemes[i]:
            transcription.append(list_of_phonemes[i])

    return transcription


def Levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_state_list():
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

    return stateList


def get_test_preprocessed(set_name):
    test_set = np.load(os.path.join(co.DATA_ROOT, 'test_preprocessed.npz'))

    test_X = test_set[set_name]
    test_Y = test_set['targets'].astype(int)

    return test_X, test_Y


def get_test_unprocessed():
    return np.load(os.path.join(co.DATA_ROOT, 'testdata.npz'))['testdata']


def get_test_sample_range(set_name):
    test_unprocessed = get_test_unprocessed()
    
    if set_name.split('_')[0] != 'dynamic':
        # start = np.random.choice(len(test_unprocessed))
        start = 290
        set = test_unprocessed[start][set_name]

    else:
        start = 290
        set = test_unprocessed[start][set_name.split('_')[1]]

    utterance = tools3.path2info(test_unprocessed[start]['filename'])[2]
    end = start+np.shape(set)[0]

    return utterance, start, end


def plot_posteriors(model, test_X, test_Y):
    utterance, one_test_sample_start, one_test_sample_end = get_test_sample_range(INPUT_KIND)

    one_test_sample = test_X[one_test_sample_start:one_test_sample_end]
    one_test_sample_targets = test_Y[one_test_sample_start:one_test_sample_end]

    posteriors = model.predict(one_test_sample)
    indices = get_predictions_as_indices(model, one_test_sample)
    phonemes = get_indices_as_phonemes(indices)

    plt.clf()
    plt.figure(figsize=(15, 10))
    for c in range(np.shape(posteriors)[1]):
        plt.plot(posteriors[:, c], label=phonemes[c])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=10, fancybox=True, shadow=True)
    plt.title('Posteriors for test sample {}, {}'.format(
                                                    one_test_sample_start,
                                                    utterance))

    if not os.path.exists(co.FIG_ROOT):
        os.mkdir(co.FIG_ROOT)

    plt.savefig(os.path.join(
                    co.FIG_ROOT,
                    '{}_predicted_posteriors_{}_{}'.format(INPUT_KIND, one_test_sample_start, utterance)))


def get_predictions_as_indices(model, samples):
    predictions = model.predict(samples)
    prediction_indices = np.argmax(predictions, axis=1)

    return prediction_indices


def get_PER(distance, reference_sequence):
    return np.divide(distance, len(reference_sequence))


def get_indices_as_phonemes(indices):
    return [stateList[i] for i in indices]


def get_nth_best_model(log, n):
    sorted_models = sorted(log, key=lambda entry: entry['test_acc'], reverse=True)

    nth_model_input = sorted_models[n]['input']
    global INPUT_KIND
    INPUT_KIND = nth_model_input

    nth_model_location = sorted_models[n]['model_location']

    nth_model = load_model(nth_model_location)

    return nth_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.clf()
    plt.figure(figsize=(20, 20))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(co.FIG_ROOT, '{}_confusion_matrix.png'.format(INPUT_KIND)))


stateList = get_state_list()

if __name__ == '__main__':
    preprocessor = Preprocessor()
    co = Constants()
    logger = Logging()

    log = logger.read_log()

    model = get_nth_best_model(log, 0)

    test_X, test_Y = get_test_preprocessed(INPUT_KIND)
    test_X = test_X[:150000, :]
    test_Y = test_Y[:150000]

    plot_posteriors(model, test_X, test_Y)

    ##############################################
    prediction_indices = get_predictions_as_indices(model, test_X)
    cnf_matrix = confusion_matrix(test_Y, prediction_indices)

    correct = len(prediction_indices) - np.count_nonzero(prediction_indices - test_Y)

    print("Correct frame-by-frame at the state level: {}, ratio: {}".format(correct, correct/len(prediction_indices)))

    plot_confusion_matrix(cnf_matrix, stateList, normalize=True,
                          title='Normalized confusion matrix')

    ##############################################

    # we get the phoneme name by the statelist
    predicted_phonemes = get_indices_as_phonemes(prediction_indices)
    actual_phonemes = get_indices_as_phonemes(test_Y)

    subed_predictions = substitute_phonemes_with_generic_name(predicted_phonemes)
    subed_actual = substitute_phonemes_with_generic_name(actual_phonemes)

    wrong = 0
    for i, _ in enumerate(subed_predictions):
        if subed_predictions[i] != subed_actual[i]:
            wrong += 1

    correct = len(subed_predictions) - wrong

    print("Correct frame-by-frame at the phoneme level: {}, ratio: {}".format(correct, correct/len(subed_predictions)))

    ##############################################

    predicted_transcription = make_transcription(predicted_phonemes)
    actual_transcription = make_transcription(actual_phonemes)

    edit_distance = Levenshtein(predicted_transcription, actual_transcription)
    PER = get_PER(edit_distance, actual_transcription)

    print("PER at the state level: ", PER)

    ##############################################

    predicted_transcription = make_transcription(subed_predictions)
    actual_transcription = make_transcription(subed_actual)

    edit_distance = Levenshtein(predicted_transcription, actual_transcription)
    PER = get_PER(edit_distance, actual_transcription)

    print("PER at the phoneme level: ", PER)
