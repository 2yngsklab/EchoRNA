# EchoRNA: Designing nucleotide sequences directly from protein topology

### Overview
EchoRNA is a discrete diffusion model that generates functional RNA sequences conditioned on a 3D structure of RNA-binding protein (RBP). This repository provides a comprehensive, easy-to-use workflow for deploying EchoRNA and sampling novel, functional RNAs. By enabling researchers to seamlessly generate RNAs that can bind to their RBPs of interest, this project aims to accelerate scientific discovery and therapeutic applications of RNA biology. 

<div align="center">
<img src="./images/PUM2sampling.gif" alt="PUM2 RNA motif generation" width="400">
</div>

*EchoRNA generating RNA sequences containing the UGUA motif known to bind Pumilio-family proteins when conditioned on PUM2 (PDB ID: 3Q0Q) structure.*

### Architecture
EchoRNA consists of three main components:

1. **Protein Encoder**: GVP-GNN with 6 layers processing residue-level graphs with ESM-2 and ESM-IF embeddings
2. **RNA Language Model**: RNA-FM with LoRA adaptations applied to attention layers
3. **Cross-attention Transformer**: Integrates RNA and protein features with timestep conditioning

<div align="center">
<img src="./images/architecture.png" alt="EchoRNA Architecture" width="600">
</div>

### Contact
For inquiries regarding the model, generated results, or technical support, please contact:

- **Daniil Melnichenko** (tordaz2000@gmail.com)  
*HITS Inc.*
- **Joohyun Cho** (joohyun98@kaist.ac.kr)  
*Korea Advanced Institute of Science and Technology (KAIST)*
- **Jongmin Lim** (jmlim2@kaist.ac.kr)  
*Korea Advanced Institute of Science and Technology (KAIST)*

### Content List
- [Dependencies and Requirements](#dependencies-and-requirements)
  - [Software](#software-dependencies)
  - [Computing system](#computing-system-requirements)
- [Installation](#installation)
  - [Environment setup](#environment-setup)
- [Usage](#usage)
  - [Protein feature engineering](#protein-feature-engineering)
  - [RNA sampling](#rna-sampling)
- [Tutorial](#tutorial)
- [Citation](#citation)
- [License](#license)

## Dependencies and Requirements
EchoRNA was developed and tested on an NVIDIA L40 GPU with the following OS and Python package dependencies
### Software dependencies
- CentOS Linux 7 with GCC 4.8.5 and Glibc 2.17
- Python 3.10.18
- torch 2.4.0+cu118 with CUDA 11.8 and CUDNN 9.1.0
- torch-geometric 2.7.0
- rna-fm 0.2.2
- faIr-esm
- transformers 4.46.3 ? 

### Computing system requirements
- CPU  
TBD
- GPU  
TBD


## Installation
### Environment setup
- esm2 smth

The project requires a conda environment with CUDA support. Use the provided environment file:

```bash
# Create environment from the provided YAML
conda env create -f echorna.yaml
conda activate rnpdiffuse

# Or create lemon environment as specified in user instructions
conda activate lemon
```


## Usage
### Protein feature engineering
### RNA sampling
- **Variable length support**: Handles RNA sequences from 8-254 nucleotides
Generate RNA sequences conditioned on protein structures:

```bash
# Sample using EchoRNA model
python sampling.py --config="sampling_config/EchoRNA.yaml"

# Sample using RF2NA baseline
python sampling.py --config="sampling_config/RF2NA.yaml"
```

## Tutorial
TBD
with AF3 structure prediction


## Citation

If you use EchoRNA in your research, please cite:

```bibtex
@article{younglab2026echorna,
  title   = {TBD},
  author  = {Melnichenko, Daniil and Cho, Joohyun and Lim, Jongmin and Yang, Sungchul
             and Back, Haeun and Kim, Dongsup and Lee, Young-suk},
  journal = {TBD},
  year    = {TBD}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
