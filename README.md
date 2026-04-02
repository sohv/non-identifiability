# On the Non-Identifiability of Steering Vectors in Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.06801-b31b1b.svg)](https://arxiv.org/abs/2602.06801) [![YouTube](https://img.shields.io/badge/YouTube-Watch-red.svg)](https://www.youtube.com/watch?v=7_pk2iE5JLo)

This repository provides empirical validation of non-identifiability in persona steering vectors for language models.

## Installation

### Setup

```bash
uv sync
```

### HuggingFace Authentication

Authenticate with HuggingFace to access both models:

```bash
huggingface-cli login
```

## Configuration

Configuration files are located in the `config/` directory:

- **`prompts.json`**: Persona prompts for all traits
- **`config.yml`**: Model configurations

## Running Experiments

**Available Traits**: `formality`, `politeness`, `sentiment`, `truthfulness`, and `agreeableness`. You can specify any combination of these traits.

**Test orthogonal component irrelevance**:
```bash
python src/experiments/test_orthogonal.py --traits formality politeness sentiment truthfulness agreeableness --n_seeds 10 --model Qwen/Qwen2.5-3B-Instruct
```

**Test alpha sweep (varying steering strength)**:
```bash
python src/experiments/alpha_sweep.py --traits formality politeness sentiment truthfulness agreeableness --alphas 0.0 0.5 1.0 2.0 --n_seeds 10 --model Qwen/Qwen2.5-3B-Instruct
```


**Test multi-environment validation**:
```bash
python src/experiments/multi_environment_validation.py --traits formality politeness sentiment truthfulness agreeableness --model Qwen/Qwen2.5-3B-Instruct
```

**Test logit distance equivalence**:
```bash
python src/experiments/logit_distance_equivalence_test.py --traits formality politeness sentiment truthfulness agreeableness
```

**Test vector equivalence (non-orthogonal)**:
```bash
python src/experiments/test_vector_equivalence.py --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B-Instruct --traits formality politeness sentiment truthfulness agreeableness
```

**Measure null-space dimensionality**:
```bash
python src/experiments/nullspace_dimensionality.py
```

**Test null-space spanning (subspace equivalence)**:
```bash
python src/experiments/nullspace_spanning.py --trait formality --n_individual_checks 50 --n_subspace_samples 5
```

## Project Structure

```
.
├── config/
│   ├── prompts.json
│   ├── config.yml
│   └── style.yaml
├── src/
│   └── experiments/
│       ├── persona_vector_experiment.py
│       ├── test_orthogonal.py
│       ├── alpha_sweep.py
│       ├── multi_environment_validation.py
│       ├── logit_distance_equivalence_test.py
│       ├── test_vector_equivalence.py
│       ├── nullspace_dimensionality.py
│       └── nullspace_spanning.py
└── data/
```

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{venkatesh2026non,
  title={On the Non-Identifiability of Steering Vectors in Large Language Models},
  author={Venkatesh, Sohan and Mahendran Kurapath, Ashish},
  journal={arXiv e-prints},
  pages={arXiv--2602},
  year={2026}
}
```

