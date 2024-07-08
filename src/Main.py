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
from RNN import forward

# properties array [max number of layers, 5 layer properties, number of simulations]
# 5 optical properties include [n, mu_a, mu_s, g, t]
# mu_a, mu_s, and t must have matching base units
prop = np.zeros((2, 5, 3))

# simulation 1: single layer
prop[0, :, 0] = [1.48, 0.01, 0.02, 0.14, 20]

# simulation 2: dual layer
# top layer
prop[0, :, 1] = [1.55, 0.83, 0.22, 0.17, 50]
# second layer
prop[1, :, 1] = [1.55, 4.22, 4.79, 0.84, 300]

# simulation 3: single layer
prop[0, :, 2] = [1.48, 20.1, 10000, 0.14, 0.01]

# run RNN
# returns array of size [number of simulations, 3]
# 3 values are R, A, T
results = forward(prop)

# print results
print("\n")
for i in range(len(results[:, 0])):
    print("Sim", i+1, "R:", np.round(results[i, 0], 4), " A:", np.round(results[i, 1], 4), " T:", np.round(results[i, 2], 4))