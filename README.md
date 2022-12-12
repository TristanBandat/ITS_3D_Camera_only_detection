# ITS_3D_Camera_only_detection

# Installation

The fastest way to install the necassary dependencies is via conda:
`conda install -f environment.yml`


## Important files and folders

### main.py
The entry point is the main.py file. Here one can find all the different hyperparameters and available models to train.
Furthermore the path to the dataset and the final models is also chosen here.
If all the necessary packages are installed one can simply run the file and the training starts.

### train.py

Here one can find the whole project structure. Further details and explanations are contained in the notebook [ITS_3D_Camera_only_challenge](ITS_3D_Camera_only_challenge.ipynb).

### results

In the results folder the final model and the plots can be found. With the help of tensorboard one can also view addtional plots like train/validation loss or the gradients.
