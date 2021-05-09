# The COBALT Algorithm for Constrained Grey-box Optimization of Computationally Expensive Models

This code was used to generate the results of a recently submitted paper titled "COBALT: COnstrained Bayesian optimizAtion of computationaLly expensive grey-box models exploiting derivaTive information". The paper is under review in Computers and Chemical Engineering and will be uploaded to arxiv shortly. If you use this code in your research, please cite this paper.

## Getting Started

To use COBALT, download all files contained in the repository and run the algorithm on the provided test problems under the 'Example' folder. To use the algorithm on your own functions, simply copy the same format as that shown in the 'main_examples.m' script. The algorithm can be applied to any number of decision variables and any number of black-box simulators as long as they are callable by a Matlab function. The objective function f and constraint functions g should be differentiable and implemented within the CasADi framework to exploit derivative information. 

## Requirements

* CasADi (https://web.casadi.org) must be installed, as we use its automatic differentiation capabilities to exploit derivative information for the known parts of the function 
* The bayesopt function (https://www.mathworks.com/help/stats/bayesopt.html), which is a part of the Statistics and Machine Learning Toolbox in Matlab, must be installed to perform some of the comparisons. 
