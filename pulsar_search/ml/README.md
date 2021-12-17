# Machine learning for pulsar detection

Machine learning methods have been used quite heavily to detect pulsars :

* https://arxiv.org/pdf/2005.01208.pdf
* https://arxiv.org/pdf/2002.08519.pdf

The dataset usually used is HTRU2 :

* https://archive.ics.uci.edu/ml/datasets/HTRU2

## K-nearest neighbors
Knn have been used to classify pulsars :

* https://www.researchgate.net/publication/321987873_Pulsar_Selection_Using_Fuzzy_knn_Classifier

There have been attempts to implement knn on quantum chips :

Using Grover to find minimum index:
* https://arxiv.org/pdf/2003.09187.pdf
* https://github.com/GroenteLepel/qiskit-quantum-knn
* https://www.ru.nl/publish/pages/913395/daniel_kok_4_maart_2021.pdf

Using the Hamming distance to parrallelize distance calculation
* https://arxiv.org/pdf/2103.04253.pdf



Can we do quantum knn for pulsar classification ?


## Kernel methods
Quantum Kernel methods with pennylane : https://pennylane.ai/qml/demos/tutorial_kernels_module.html
Quantum Kernel methods with Qiskit : https://qiskit.org/documentation/machine-learning/tutorials/03_quantum_kernel.html

## Neural Network

Neural Nets have been used for pulsar detection : 
SPINN: a straightforward machine learning solution to the pulsar candidate selection problem
https://arxiv.org/abs/1406.3627

And of course we can implement some Neural Net on quantum computers wiht Qiskit:
https://qiskit.org/documentation/machine-learning/tutorials/01_neural_networks.html
https://qiskit.org/documentation/machine-learning/tutorials/02_neural_network_classifier_and_regressor.html

or with Pennylane :
https://pennylane.ai/qml/glossary/quantum_neural_network.html
https://pennylane.ai/qml/demos/learning2learn.html



