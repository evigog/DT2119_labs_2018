import numpy as np
import proto2

data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

modellist = {}
for digit in prondict.keys():
    modellist[digit] = ['sil'] + prondict[digit] + ['sil']

wordHMMs['o'] = concatHMMs(phoneHMMs, modellist['o'])