# Continual-SDDAL
The continual learning in Simulation-Driven Differentiable Active Learning framework for efficient data generation

In the initial commit to this Continual-SDDAL repository, an original SDDAL (Simulation-Driven Differentiable Active Learning) framework was pushed into this repository. The following instructures are about how to run different functionalities in the initially committed SDDAL. After the continual researcher is familiar with the initially committed original SDDAL, she will proceed to develop an continual learning scheme inside the SDDAL framework based on the initially pushed SDDAL code base.

# Folder structure preparation

Before running any experiments, the following preparations must be made:

1, Establish 6 folders "Design_rec", "Design_ring", "Design_chair", "Design_gaussian", "Design_hat", and "Design_tear" with the same subfolder structure inside:

----Design_rec

   --latest_uncertainty

--|

   --models

# Active dataset design for one of the 6 beam shapes, e.g., RecTophat shape
