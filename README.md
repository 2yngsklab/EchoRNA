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
  - [Downloading Model Weight](#downloading_model_weight)
- [Usage](#usage)
  - [Required Arguments](#required_arguments)
  - [Optional Arguments](#optional_arguments)
- [Tutorial](#tutorial)
- [Output Directory Structure](output_directory_structure)
- [Citation](#citation)
- [License](#license)

## Dependencies and Requirements

EchoRNA was developed and tested on an NVIDIA L40 GPU with the following OS and Python package dependencies

### Software dependencies
- CentOS Linux 7 with GCC 4.8.5 and Glibc 2.17
- Python 3.10.18
- torch 2.4.0+cu118 with CUDA 11.8 and CUDNN 9.1.0
- torch-geometric 2.7.0
- trnasformer 4.46.3
- rna-fm 0.2.2
- faIr-esm 2.0.0 with biotite 1.2.0

### Computing system requirements
- CPU  
TBD
- GPU  
TBD


## Installation

### Environment setup
Run ./install.sh to set up the Conda environment for EchoRNA.  
You can customize the installation directory and environment name using the --install-dir and --env-name flags.  

```bash
# Create the default Conda environment ('echorna')
./install.sh

# Create a Conda environment with a custom name and directory
./install.sh --install-dir=<INSTALL_DIR> --env-name=<ENV_NAME>
```

Once installed, activate the environment before running the pipeline  

```bash
conda activate echorna     # If you used the default name
conda activate <ENV_NAME>  # If you specified a custom name
```

(Optional) To verify that your GPU is available for RNA sampling, run the following command  

```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Downloading Model Weight
TBD

## Usage

```
mv echorna_weight.pth ./EchoRNA  # move model weight to the working directory
python ./EchoRNA/EchoRNA_sampling.py --protein <PROTEIN> --chain <CHAIN> --output-dir <OUTPUT_DIR> [--name <OUTPUT_FILE_NAME>] [--rna-length <LENGTH>] [--num-sequence <NUMBER>] [--sampling-strategy <SAMPLING_STRATEGY>] [--random-seed <RANDOM_SEED>] [--config <CONFIG_PATH>] [''weight <WEIGHT_PATH>] [--GPU]
```

### Required Arguments

- `-p`, `--protein` : Path to the protein structure. It should be CIF format.  
- `-c`, `--chain` : The ID of protein chain where generated RNA binds.
- `-d`, `--output-dir` : The output directory where generated sequences are saved

### Optional Arguments

- `-n`, `--name` : The namd of output file. If not provided, root namd of CIF file will be used.
- `-l`, `--rna-length` : The length of RNA to be generated (default: `20`).
- `-ns`, `--num-sequence` : The number of RNAs to be generated (default: `20`).
- `-s`, `--sampling-strategy` : Sampling strategy. You can select between `vanilla` and `gumbel_argmax` (default: `vanilla`).
- `-sd`, `--random-seed` : Random seed (default: `42`).
- `-g`, `--GPU` : Use GPU if available. If not provided, CPU will be used.
- `--config` : Path to model configuration file. (default: "./echorna_config.yaml").
- `--weight` : Path to the model weight. (default: "./echorna_weight.pth").

## Tutorial

TBD


## Output Directory Structure
```
output_dir/
|-- complex/      # Proteinn structure files 
|-- esm2/         # Esm2 embedding files
|-- esmIF/        # EsmIF embedding files
|-- RNA/          # EchoRNA-generated RNAs
```

## Citation

If you use EchoRNA in your research, please cite:

```bibtex
@article{2yngsklab2026echorna},
  title   = {TBD},
  author  = {Melnichenko, Daniil and Cho, Joohyun and Lim, Jongmin and Yang, Sungchul
             and Back, Haeun and Kim, Dongsup and Lee, Young-suk},
  journal = {TBD},
  year    = {TBD}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
