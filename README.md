# EchoRNA: Designing nucleotide sequences directly from protein topology

### Overview
EchoRNA is a discrete diffusion model that generates functional RNA sequences conditioned on a three-dimensional structure of RNA-binding protein.

<div align="center">
<img src="./images/PUM2sampling.gif" alt="PUM2 RNA motif generation" width="400">
</div>

*EchoRNA generating RNA sequences containing the UGUA motif known to bind Pumilio-family proteins when conditioned on PUM2 (PDB ID: 3Q0Q) structure.*

### Architecture
EchoRNA consists of three main components:

1. **Protein Encoder**: GVP-GNN with 6 layers processing residue-level graphs with ESM-2 (1280D) and ESM-IF (512D) embeddings
2. **RNA Language Model**: RNA-FM with LoRA adaptations (r=64, α=128) applied to attention layers
3. **Cross-attention Transformer**: Integrates RNA and protein features with timestep conditioning

<div align="center">
<img src="./images/architecture.png" alt="EchoRNA Architecture" width="600">
</div>


### Contact
### Content List

## Requirements
EchoRNA is build with these packages
### Software
- os, torch, cuda, cudnn, torch-geometric, esm2, esmif, rnafm, etc
### Computing system
- CPU
- GPU




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

### Key Dependencies

- PyTorch 2.4.0+cu118 with CUDA 11.8
- torch-geometric 2.5.3
- transformers 4.46.3
- fair-esm 2.0.0
- RNA-FM
- BioPython, pandas, numpy, matplotlib
- Rich (for terminal formatting)





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

## Examples
TBD
with AF3 structure prediction


## Citation

If you use EchoRNA in your research, please cite:

```bibtex
@article{melnichenko_cho2025echorna,
  title   = {Topological Transcription for Protein-Binding RNA Sequence Design via Discrete Diffusion},
  author  = {Melnichenko, Daniil and Cho, Joohyun and Yang, Sungchul and Lim, Jongmin
             and Back, Haeun and Kim, Dongsup and Lee, Young-suk},
  journal = {bioRxiv},
  year    = {2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
