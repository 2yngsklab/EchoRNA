# EchoRNA: Designing nucleotide sequences directly from protein topology

### Overview
EchoRNA is a discrete diffusion model that generates functional RNA sequences conditioned on the 3D structure of an RNA-binding protein (RBP). This repository provides a streamlined, easy-to-use workflow for generating novel, functional RNA sequences with EchoRNA. The goal of this project is to accelerate scientific discovery and therapeutic development in RNA biology.

<div align="center">
<img src="./images/PUM2sampling.gif" alt="PUM2 RNA motif generation" width="400">  
</div>

<div align="center">
<i>EchoRNA generating RNA sequences containing a PUM-specific binding motif (UGUA) when conditioned on PUM2 (PDB ID: 3Q0Q).</i>
</div>

### Architecture
EchoRNA consists of three main components:

1. **Protein Encoder**: A 6-layer GVP-GNN that encodes protein structure from residue-level graphs using ESM-2 and ESM-IF embeddings.
2. **RNA Encoder**: A LoRA-adapted RNA-FM that encodes RNA sequences.
3. **Cross-attention Transformer**: An 8-layer decoder that uses the RNA sequence representation as queries (Q) and the protein graph representation as keys (K) and values (V).

<div align="center">
<img src="./images/architecture.png" alt="EchoRNA Architecture" width="600">
</div>

### Contact
For inquiries regarding the model, generated results, or technical support, please contact:

- **Joohyun Cho** (joohyun98@kaist.ac.kr)  
*Korea Advanced Institute of Science and Technology (KAIST)*
- **Sungchul Yang** (pacific333@kaist.ac.kr)  
*Korea Advanced Institute of Science and Technology (KAIST)*  

### Content List 
- [Dependencies and Requirements](#dependencies-and-requirements)
  - [Software dependencies](#software-dependencies)
  - [Computing system requirements](#computing-system-requirements)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Set up the conda environment](#set-up-the-conda-environment)
  - [Download Model Weight](#download-model-weight)
- [Usage](#usage)
  - [Required Arguments](#required-arguments)
  - [Optional Arguments](#optional-arguments)
- [Tutorial](#tutorial)
  - [Output directory](#output-directory)
- [Other Details](#other-details)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Dependencies and Requirements
EchoRNA requires [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) for dependency management (`echoRNA_env.yaml`). The model was trained on a single NVIDIA L40 GPU using the dependencies listed below.

### Software dependencies
- CentOS Linux 7 with GCC 4.8.5 and Glibc 2.17
- Python 3.10.18
- torch 2.4.0+cu118 with CUDA 11.8 and cuDNN 9.1.0
- torch-geometric 2.7.0
- transformers 4.46.3
- rna-fm 0.2.2
- fair-esm 2.0.0
- biotite 1.2.0

### Computing system requirements
- EchoRNA requires at least 8 GB of free disk space to store the pretrained model weights.  
- For GPU use, check the [CUDA](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) and [PyTorch](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) versions compatible with your [NVIDIA GPU architecture](https://developer.nvidia.com/cuda/gpus). Note that NVIDIA Blackwell GPUs require torch >= 2.7.
  
**Handling an OOM (out-of-memory) Error**
- The system memory and VRAM required by EchoRNA vary depending on the input protein length and the number and length of the RNA sequences to be generated. If you hit an OOM error, try reducing the number of sequences generated per run (`-n`) or the sequence length (`-l`).
- For reference, running EchoRNA with 3Q0Q on CPU used 7 GB of system memory, and on GPU used 6 GB of VRAM.

## Installation
### Clone the Repository
Download the source code to your machine and navigate into the project directory.

```bash
git clone https://github.com/2yngsklab/EchoRNA.git  
cd EchoRNA # the project directory
```

### Set up the conda environment
Run ```install.sh``` to set up the Conda environment for EchoRNA. You can customize the environment name using the ```--envname``` flag.  

```bash
chmod +x ./install.sh

# Create the default Conda environment ('echorna')
./install.sh

# Create a Conda environment with a custom name
./install.sh --envname=<ENV_NAME>
```

Once installed, activate the environment before running the pipeline  

```bash
conda activate echorna     # If you used the default name
conda activate <ENV_NAME>  # If you specified a custom name
```

(Optional) To verify GPU availability, run the following command.  

```python
python -c "import torch; print(torch.cuda.is_available())"
>>> True (if GPU is available)
```

### Download Model Weight
You can download the model weight from [Hugging Face](https://huggingface.co/2yngsklab/EchoRNA)

```bash
hf download 2yngsklab/EchoRNA echorna_weight.pth --local-dir .
```

Pretrained weights for the RNA and protein foundation models can be downloaded via Python.

```python
import fm, esm
fm.pretrained.rna_fm_t12() # RNA-FM
esm.pretrained.esm2_t33_650M_UR50D() # ESM-2
esm.pretrained.esm_if1_gvp4_t16_142M_UR50() # ESM-IF
```

If the RNA-FM download fails, try fetching it directly.
```bash
hf download cuhkaih/rnafm RNA-FM_pretrained.pth --local-dir ~/.cache/torch/hub/checkpoints/
```

## Usage
```
python sample_EchoRNA.py -p <STRUCTURE> -c <CHAIN> \
    [-d <OUTPUT_DIR>] \
    [-fn <FILE_NAME>] \
    [-n <NUM_SEQUENCE>] \
    [-l <RNA_LENGTH>] \
    [-s <SAMPLING_STRATEGY>] \
    [-sd <RANDOM_SEED>] \
    [-g [<DEVICE>]] \
    [--config <CONFIG_PATH>] \
    [--weight <WEIGHT_PATH>]
```

### Required Arguments

- `-p`, `--protein` : Input structure file in [CIF](https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax) format.  
- `-c`, `--chain` : Protein chain ID to use.  

### Optional Arguments
- `-d`, `--output-dir` : Directory where the generated sequences are saved (default: `./output`).
- `-fn`, `--name` : Name of the output file (default: `name of the CIF file`).
- `-n`, `--num-sequence` : Number of RNA sequences to generate (default: `100`).
- `-l`, `--rna-length` : Length of RNA sequences to generate (default: `20`).
- `-s`, `--sampling-strategy` : Sampling strategy; choose between `vanilla` and `gumbel` (default: `'vanilla'`).
- `-sd`, `--random-seed` : Random seed (default: `42`).
- `-g`, `--GPU` : Enable GPU usage. If this flag is omitted, CPU will be used. When enabled, you can optionally specify a device `cuda:<number>` (default: `'cuda:0'`).
- `--config` : Model configuration file (default: `'./echorna_config.yaml'`).
- `--weight` : Model weight file (default: `'./echorna_weight.pth'`).

## Tutorial
This tutorial generates RNA sequences using EchoRNA for chain A of PUM2 (PDB ID: [3Q0Q](https://www.rcsb.org/structure/3Q0Q)).  
You can run this tutorial directly in your browser using the Colab notebook  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2yngsklab/EchoRNA/blob/main/tutorial/sample_EchoRNA_colab.ipynb)  
Follow the steps below to run it locally via the command line.

```bash
# Run on CPU
python sample_EchoRNA.py -p ./tutorial/3Q0Q.cif -c A
>>> Device: cpu
>>> Loading diffusion model and generating 100 RNA sequences of length 20
>>> Wrote 100 RNA sequences to ./output/RNA/3Q0Q.fasta
```

If a GPU is available, you can run EchoRNA on it instead.
```bash
# Run on GPU
python sample_EchoRNA.py -p ./tutorial/3Q0Q.cif -c A -fn 3Q0Q_gpu -g
# Run on the second GPU
# python sample_EchoRNA.py -p ./tutorial/3Q0Q.cif -c A -fn 3Q0Q_gpu -g cuda:1 
>>> Device: cuda:0
>>> Loading diffusion model and generating 100 RNA sequences of length 20
>>> Wrote 100 RNA sequences to ./output/RNA/3Q0Q_gpu.fasta
```

You can also generate any number of RNA sequences of a desired length:
```bash
# Generate 50 EchoRNAs of length 8
python sample_EchoRNA.py -p ./tutorial/3Q0Q.cif -c A -fn 3Q0Q_50_8 -n 50 -l 8
>>> Device: cpu
>>> Loading diffusion model and generating 50 RNA sequences of length 8
>>> Wrote 50 RNA sequences to ./output/RNA/3Q0Q_50_8.fasta
```

### Output directory
After running the tutorial, the output directory will contain the following subdirectories

```
./output/
|-- complex/      # Parsed protein structures (.pickle)
|-- esm2/         # ESM-2 sequence embeddings (.pt)
|-- esmIF/        # ESM-IF inverse-folding embeddings (.pt)
|-- RNA/          # Generated RNA sequences (.fasta)
```

Your generated RNA sequences will be written to `./output/RNA/<name>.fasta`.  
The other subdirectories hold intermediate files that EchoRNA caches and reuses, so repeated runs on the same protein skip recomputation.

## Other Details
**Echo Dataset**
- The `./dataset` directory contains the PDB IDs and corresponding protein and RNA chain identifiers for the high-quality, non-redundant protein–RNA complex pairs used in the training, validation, and test sets.  


**Generating from an AF3-predicted structure**  
- AlphaFold3 provides a [web interface](https://alphafoldserver.com/) that predicts the structures of biomolecules (proteins and nucleic acids) from their sequences. See the [AF3 server guidelines](https://alphafoldserver.com/guides) for details.
- EchoRNA can generate RNA sequences directly from an AF3-predicted protein structure.
```bash
# Run EchoRNA with AF3-predicted PUM2 structure
python sample_EchoRNA.py -p ./tutorial/3Q0Q_af3.cif -c A
>>> Device: cpu
>>> Loading diffusion model and generating 100 RNA sequences of length 20
>>> Wrote 100 RNA sequences to ./output/RNA/3Q0Q_af3.fasta
```

## Acknowledgement
We would like to thank [RNAflow](https://github.com/divnori/rnaflow), whose work was helpful during the development of this project.


## Citation

If you use EchoRNA in your research, please cite:

```bibtex
@article{2yngsklab2026echorna,
  title   = {Designing functional RNA sequences directly from protein topology with EchoRNA},
  author  = {Melnichenko, Daniil and Cho, Joohyun and Lim, Jongmin and Yang, Sungchul
             and Back, Haeun and Cho, Hyeonggon and Kim, Dongsup and Lee, Young-suk},
  journal = {TBD},
  year    = {TBD},
  doi     = {TBD}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
