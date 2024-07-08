# RNN-MC
Recurrent Neural Network (RNN) for accelerated multi-layer Monte Carlo predictions
# Authors and Citation
- Daniel Carne: dcarne@purdue.edu
- Ziqi Guo: gziqi@purdue.edu
- Xiulin Ruan: ruan@purdue.edu

Cite:
# How to use
The src folder contains two python files and a folder titled 'RNN_Model'. To run RNN-MC, open the Main.py file where comments detail how to run the program. The Main.py file, RNN.py file, and RNN_Model folder must all be located in the same directory. The RNN_Model folder contains all the weights and biases for the RNN.

The user will create and edit the array titled "prop" in the Main.py file which contains three dimensions [maximum number of layers to simulate, optical properties, number of simulations]. The number of optical properties will always be 5, which are the refractive index (n), absorption coefficient (mu_a), scattering coefficient (mu_s), asymmetry parameter (g), and thickness (t). The base units for the absorption coefficient, scattering coefficient, and thickness must be the same.
# About
The RNN here is trained on 224,000 plane-parallel Monte Carlo simulations across a broad range of optical properties to provide up to 1000x speedup.
