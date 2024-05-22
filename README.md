# RC-DiffFlow

This repository is the official implementation of 
[RC-DiffFlow: A Robust Framework for Dimensionality Reduction and Kinetic Modeling in Molecular Dynamics]

Our implementation is built upon pytorch. 

## Codebase structure

For each molecule evaluated in the paper we provide a separate test file (both in general .py and notebook format). These generally follow the same structure. 

- /cfg: Contains hyper-parameters.
- /models: Implementation of the models of Data-Depentdent-Diffusion-Model, Normalizing Flow and GMM. 
- /utils: tools for constructing dataset, training block and some visualization tool. 