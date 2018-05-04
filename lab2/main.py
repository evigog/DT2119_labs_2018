import numpy as np
import proto2
import prondict
import tools2
import matplotlib.pyplot as plt

#score utterances using viterbi
def test_viterbi(data, wordHMMs):
    # Score all utterances using viterbi
    viterbi_res = []
    for u in range(data.size):
        u_hmm = []
        for hmm in wordHMMs.keys():
            # compute log emission probability of utterance u and model hmm
            log_prob = tools2.log_multivariate_normal_density_diag(data[u]['lmfcc'], wordHMMs[hmm]['means'],
                                                            wordHMMs[hmm]['covars'])
            # viterbi
            res = proto2.viterbi(log_prob, wordHMMs[hmm]['startprob'], wordHMMs[hmm]['transmat'])
            res['label'] = hmm
            u_hmm.append(res)

        viterbi_res.append(u_hmm)

    # assign each utterance to hmm with max likelihood and compute error
    count_error = 0
    for i in range(data.size):
        hmm_list = viterbi_res[i]
        hmms_loglik = [hmm['loglik'] for hmm in hmm_list]
        max_idx = hmms_loglik.index(max(hmms_loglik))
        if (data[i]['digit'] != hmm_list[max_idx]['label']):
            count_error += 1

    print('Number of missclassified utterances - Viterbi: ', count_error)


def get_alpha_log_likelihood(alphas):
    return tools2.logsumexp(alphas[-1, :])


def score_utterances_by_forward_probs(data, wordHMMs):
    scores = {}
    for datapoint in data:
        digit = datapoint['digit']
        features = datapoint['lmfcc']

        per_model_scores = []
        for word_hmm in wordHMMs.items():
            means = word_hmm[1]['means']
            covars = word_hmm[1]['covars']
            transmat = word_hmm[1]['transmat']
            startprob = word_hmm[1]['startprob']

            log_emlik = tools2.log_multivariate_normal_density_diag(features,
                                                                    means,
                                                                    covars)

            forward_prob = proto2.forward(log_emlik, startprob, transmat)

            score = get_alpha_log_likelihood(forward_prob)
            per_model_scores.append((word_hmm[0], score))

        # Keep the one with the highest score
        highest_score_model = sorted(per_model_scores, key=lambda d: d[1], reverse=True)[0]
        scores[digit] = highest_score_model

    scorrings = [digit == score_tup[0] for digit, score_tup in scores.items()]

    return scorrings


def question_5_2(data, wordHMMs):
    d = data[10]

    loglik_list = []

    for digit in wordHMMs.keys():

        model = wordHMMs[digit]
        means = model['means']
        covars = model['covars']

        log_lik_old = 0
        for it in range(20):
            log_emission = tools2.log_multivariate_normal_density_diag(d['lmfcc'], means, covars)

            alphas = proto2.forward(log_emission, model['startprob'], model['transmat'])
            betas = proto2.backward(log_emission, model['startprob'], model['transmat'])
            gammas = proto2.statePosteriors(alphas, betas)

            log_lik = get_alpha_log_likelihood(alphas)

            print('likelihood=', log_lik, ' at iteration=', it)
            print('mean=', np.mean(means), 'covar=', np.mean(covars))
            print('\n')

            if (abs(log_lik - log_lik_old) < 1e-4):
                print('Converged!')
                break

            result = proto2.updateMeanAndVar(d['lmfcc'], gammas, varianceFloor=5.0)
            means = result['mean']
            covars = result['covar']

            log_lik_old = log_lik

        loglik_list.append(log_lik)

    return loglik_list


data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
example = np.load('lab2_example.npz')['example'].item()


modellist = {}
for digit in prondict.prondict.keys():
    modellist[digit] = ['sil'] + prondict.prondict[digit] + ['sil']

# produce HMM for each model in modelist
wordHMMs = {}
for word in modellist.keys():
    h = proto2.concatHMMs(phoneHMMs, modellist[word])
    wordHMMs[word] = h


obsloglik = example['obsloglik']
startprob = wordHMMs['o']['startprob']
transmat = wordHMMs['o']['transmat']

# forward algorithm for hmm 'o'
forward_prob = proto2.forward(obsloglik, startprob, transmat)

# fig = plt.figure()
# ax = plt.subplot(121)
# ax.set_title('Actual')
# plt.pcolormesh(example['logalpha'])

# ax = plt.subplot(122)
# ax.set_title('Computed')
# plt.pcolormesh(forward_prob)
# plt.colorbar()

# plt.savefig("Fig/forward_probs.png")

# hmms_loglik = get_alpha_log_likelihood(forward_prob)
# print("Expected log likelihood: {}, Computed: {}".format(example['loglik'],
#                                                          hmms_loglik))

scorring = score_utterances_by_forward_probs(data, wordHMMs)
unique, counts = np.unique(scorring, return_counts=True)
print(unique, counts)
print()


# backward algorithm for hmm 'o'
backward_prob = proto2.backward(example['obsloglik'], startprob, transmat)
diff = np.absolute(backward_prob - example['logbeta'])
print('Backward max difference with example: ', np.max(diff))

# viterbi for hmm 'o'
vit_result = proto2.viterbi(example['obsloglik'], startprob, transmat)
vit_dif = vit_result['loglik'] - example['vloglik'][0]
print('Viterbi difference with example: ', vit_dif)

# plot viterbi path
plt.clf()
plt.pcolormesh(forward_prob.T)
plt.plot(vit_result['path'])
plt.colorbar()
plt.savefig('Fig/viterbi_path')
plt.show()

# calculate state posteriors for wordHMMs['o']
gamma_prob = proto2.statePosteriors(forward_prob, backward_prob)

# score all utterances using viterbi
test_viterbi(data, wordHMMs)

#compute log-likelihood using expectation maximization
log_link_list = question_5_2(data, wordHMMs)
