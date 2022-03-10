## DSC-Capstone-Result-Replication
### DSC180A Quarter 1 and 2

Website code is included in the 'docs' directory of this repo.

To run the GENConv GNN model, create docker pod using jmduarte/capstone-particle-physics-domain:latest and group key to access data folder.

Then clone this repo and cd into root folder.

In the root folder, you can use python run.py [train, test] to produce a dataframe with predictions and other data useful for visualization in the src/analysis/evaluation.ipynb notebook. 

To visualize, you can simply run all cells in the eval. notebook.

Pls. ignore the loading bar in terminal it doesn't mean anything - but training it will take a while. 

There is really no need to run the training since the testing script loads presaved weights (stored in github) to initialize the pre-trained model for testing.  
