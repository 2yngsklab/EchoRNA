# EchoRNA
EchoRNA is a discrete diffusion based model that generates functional RNA seqeunces conditioned on protein graphs.

<img src="./images/PUM2sampling.gif" alt="Pumilio RNA moyif generation" width="400">
EchoRNA samples RNA sequences of any desired length that contain the binding motif. The below example shows how the resulting seqeunces contain `UGUA` motif known to bind to Pumilio-family porteins when conditioned on PUM2 (PDB id: 3q0q) structure.



to run python script train.py, you need:
`python train.py --congif="CONFIG FILE LOC"`
or just run the notebook train.ipynb