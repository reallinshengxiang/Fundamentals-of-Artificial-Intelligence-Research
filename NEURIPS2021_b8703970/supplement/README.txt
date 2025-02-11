DEPENDENCIES
The easisest way to install dependencies is to run
pip install -r requirements.txt

If pip install -r requirements fails you can install the following packages manually:
pip install numpy
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib
pip install scipy
pip install sklearn
pip install git OR conda install git
pip install git+https://github.com/ahwillia/tensortools

for more details on how to install PyTorch, see: https://pytorch.org/
The installation command varries based on OS.

MODEL TRAINING
To train an RNN model on the CDI-task, run the following:
python train_models.py XXXX_name -v 0.5 --N 2
where XXXX must be either 'bptt', 'ga', 'ff'. Note for Hebbian models we used to Github repo previously published at: https://github.com/ThomasMiconi/BiologicallyPlausibleLearningRNN We followed the directions found in this repo and just manually changed input variance values as desired in the code.

alternatively, to train on the RDM-task,
python train_models.py XXXX_model_name -v 0.5 --rdm

where model_name can be any string designating the name to save your model after training. -v 0.5 specifies to train the model with an input variance of 0.5. You can also try setting -v to 0.1 or 1.0.
--N 2 specifies the CDI-task while --rdm specifies the RDM-taks.
Training on the N-CDI tasks is also possible by specifying a value of N greater than 2. convergence is not guarenteed for N greater than or equal to 7.



VISUALIZING ATTRACTOR STATES
To visualize the attractor states and PC trajectories for a specific model (which must be contained in the models directory), and run:
python analyze_dynamics.py model_name large<br> using the included sample models, you can run <br> python analyze_dynamics.py bptt_0050 large

To view the attractors near zero, run
python analyze_dynamics.py bptt_0050 sparse



CLUSTERING RNNs
To cluster RNNs based on attractor topologies or representational geometries you will need a text file listing all the models to be analyzed. The text file should be formatted such that each line in the file corresponds to a group of RNNs (e.g. each line could correspond to a different learning rule). Then each model name on that line should be seperated by spaces. This text file must be saved in the models directory.

If you were comparing 5 BPTT and 5 GA models this might look like:
BPTT bmodel_name_1 bmodel_name_2 bmodel_name_3 bmodel_name_4 bmodel_name_5
GA gmodel_name_1 gmodel_name_2 gmodel_name_3 gmodel_name_4 gmodel_name_5

A sample text file comparing 2 BPTT RNNs to 2 GA RNNs trained on the RDM task is included. To analyze attractor topologies run,
python fixed_point_input_topology.py sample.txt

Note: to analyze the attractor topologies for CDI tasks, the N_level must be manually set in fixed_point_input_topology.py.  It is set to 1 (for the RDM task) by default.

To analyze representational geometry clusters run,
python svcca_analysis.py sample.txt -reload
Note: after running once, the -reload keyword argument can be ommitted.