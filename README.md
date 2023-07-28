NeuralFIM
==============================

NeuralFIM is a neural network module for computing a Riemannian metric (fisher information metric) on point cloud data. In this repository we apply neuralFIM to point cloud data by visualizing PHATE embeddings of toy and single-cell datasets colored by the FIM trace and volume and use learned metrics fo find geodesics on the point cloud data. 

An example notebook for generating NeuralFIM embeddings exists [here](https://github.com/guillaumehu/phate_fim/blob/main/notebooks/df-test-neuralFIM-v3.0.ipynb).
An example notebook for finding geodesics using NeuralFIM exists [here](https://github.com/guillaumehu/phate_fim/blob/main/notebooks/df_swiss_roll_geodesic.ipynb).
To train neuralFIM and generate FIMs on point cloud data via the command line, navigate to ~/phate_fim/src/models type the command: "python train_model_tree.py". For additional arguments see the script located at: ~/phate_fim/src/models/train_model_tree.py

The associated publication can be found here: [here](https://arxiv.org/abs/2306.06062).

Make sure you utilize the NeuralFIM.yml file to ensure package compatibility. 
