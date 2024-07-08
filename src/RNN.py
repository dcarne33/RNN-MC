#  Recurrent Neural Network for accelerated multi-layer Monte Carlo predictions
#  Copyright (C) 2024 Daniel Carne <dcarne@purdue.edu>
#  Copyright (C) 2024 Ziqi Guo <gziqi@purdue.edu>
#  Copyright (C) 2024 Xiulin Ruan <ruan@purdue.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numba import njit, prange
import time


@njit(parallel=True)
def rnn(b1, b2, b3, bo, bh, w1, w2, w3, wo, wh, features, predict):

    for sim in prange(len(features[0, 0, :])):
        n1 = np.zeros(len(w1[:, 0]))
        n2 = np.zeros(len(w2[:, 0]))
        n3 = np.zeros(len(w3[:, 0]))
        n4 = np.zeros(len(wo[:, 0]))
        input = np.zeros(20)
        num_layers = int(features[0, 4, sim])
        hidden_state = np.zeros(16)
        for layer in range(num_layers):
            n1[:] = 0
            n2[:] = 0
            n3[:] = 0
            n4[:] = 0
            input[:4] = features[layer, :4, sim]
            input[4:] = hidden_state[:]
            hidden_state[:] = 0
            # layer 1
            for k in range(len(w1[0, :])):
                n1[:] += input[k] * w1[:, k]
            n1[:] += b1[:]
            for j in range(len(n1)):
                if n1[j] < 0:
                    n1[j] = 0
            # layer 2
            for k in range(len(w2[0, :])):
                n2[:] += n1[k] * w2[:, k]
            n2[:] += b2[:]
            for j in range(len(n2)):
                if n2[j] < 0:
                    n2[j] = 0
            # layer 3
            for k in range(len(w3[0, :])):
                n3[:] += n2[k] * w3[:, k]
            n3[:] += b3[:]
            for j in range(len(n3)):
                if n3[j] < 0:
                    n3[j] = 0
            # output layer
            for k in range(len(wo[0, :])):
                n4[:] += n3[k] * wo[:, k]
            n4[:] += bo[:]
            # hidden state
            for k in range(len(wh[0, :])):
                hidden_state[:] += n3[k] * wh[:, k]
            hidden_state[:] += bh[:]
        predict[sim, :] = n4[:]

    return predict


def remove_ones_zeros(results):
    # sets any negative values to 0
    for i in range(len(results[:, 0])):
        if results[i, 0] < 0:
            results[i, 0] = 0
        if results[i, 1] < 0:
            results[i, 1] = 0
        if results[i, 2] < 0:
            results[i, 2] = 0
        if results[i, 0] > 1:
            results[i, 0] = 1
        if results[i, 1] > 1:
            results[i, 1] = 1
        if results[i, 2] > 1:
            results[i, 2] = 1
    return results


def forward(prop):
    # import weights and biases
    w1 = np.loadtxt('RNN_Model/w1_rnn1.7.txt', dtype=np.float32)
    w2a = np.loadtxt('RNN_Model/w2a_rnn1.7.txt', dtype=np.float32)
    w2b = np.loadtxt('RNN_Model/w2b_rnn1.7.txt', dtype=np.float32)
    w2 = np.concatenate((w2a, w2b), axis=0)
    w3a = np.loadtxt('RNN_Model/w3a_rnn1.7.txt', dtype=np.float32)
    w3b = np.loadtxt('RNN_Model/w3b_rnn1.7.txt', dtype=np.float32)
    w3 = np.concatenate((w3a, w3b), axis=0)
    wo = np.loadtxt('RNN_Model/wo_rnn1.7.txt', dtype=np.float32)
    wh = np.loadtxt('RNN_Model/wh_rnn1.7.txt', dtype=np.float32)

    b1 = np.loadtxt('RNN_Model/b1_rnn1.7.txt', dtype=np.float32)
    b2 = np.loadtxt('RNN_Model/b2_rnn1.7.txt', dtype=np.float32)
    b3 = np.loadtxt('RNN_Model/b3_rnn1.7.txt', dtype=np.float32)
    bo = np.loadtxt('RNN_Model/bo_rnn1.7.txt', dtype=np.float32)
    bh = np.loadtxt('RNN_Model/bh_rnn1.7.txt', dtype=np.float32)

    # non-dimensionalize
    prop[:, 1, :] *= prop[:, 4, :]
    prop[:, 2, :] *= prop[:, 4, :]

    # get num layers
    for i in range(len(prop[0, 0, :])):
        num_layers = len(prop[:, 0, 0])
        for j in range(len(prop[:, 0, 0])):
            if prop[j, 0, i] == 0:
                num_layers = j
                break
        prop[:, 4, i] = num_layers

    # normalize each feature
    prop[:, 0, :] = (prop[:, 0, :]) / 2.5
    prop[:, 1, :] = (np.log10(prop[:, 1, :] + 0.0001)+4)/8.69154
    prop[:, 2, :] = (np.log10(prop[:, 2, :] + 0.0001)+4)/8.69549
    prop[:, 3, :] = -((-prop[:, 3, :]+1)**(1/3)) + 1

    # Initialize and compile RNN
    print("Compiling RNN...")
    temp_prop = np.ones((1, 5, 1))
    predict = np.zeros((len(prop[0, 0, :]), 3))
    predict = rnn(b1, b2, b3, bo, bh, w1, w2, w3, wo, wh, prop, predict)


    # run RNN
    print("Running RNN")
    predict = np.zeros((len(prop[0, 0, :]), 3))
    start = time.time()
    predict = rnn(b1, b2, b3, bo, bh, w1, w2, w3, wo, wh, prop, predict)
    end = time.time()
    print("RNN finished    Time:", np.round(end-start, 4), "seconds")

    # normalize prediction
    predict = remove_ones_zeros(predict)
    predict[:, 0] = (predict[:, 0] / (predict[:, 0] + predict[:, 1] + predict[:, 2]))
    predict[:, 1] = (predict[:, 1] / (predict[:, 0] + predict[:, 1] + predict[:, 2]))
    predict[:, 2] = (predict[:, 2] / (predict[:, 0] + predict[:, 1] + predict[:, 2]))
    return predict