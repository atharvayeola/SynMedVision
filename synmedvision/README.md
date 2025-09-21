# SynMedVision

SynMedVision provides a reproducible reference pipeline for generating, evaluating, and serving synthetic histopathology imagery derived from the PatchCamelyon (PCam) dataset.  The repository contains deterministic data preparation utilities, light-weight baseline training loops for StyleGAN2-ADA and diffusion models with LoRA fine-tuning, and a collection of evaluation and reporting scripts that assess realism, diversity, downstream utility, and privacy risks.

The implementation in this repository focuses on providing an end-to-end skeleton that is fully testable inside a lightweight development container.  Heavy weight training and evaluation logic is abstracted behind clear interfaces so that the core research workflow can be extended with high fidelity models when operating on GPU equipped infrastructure.

## Repository Layout

```
.
├── configs/                # YAML configuration files for all CLI entrypoints
├── docker/                 # Container recipe for GPU execution
├── src/                    # Implementation modules (data, models, eval, cli, demo)
├── tests/                  # Unit tests exercising the lightweight pipeline
├── data/, out/, reports/   # Runtime directories populated by the CLI utilities
└── Makefile                # Convenience commands mirroring the execution plan
```

## Quick Start

1. **Create a virtual environment and install dependencies**
   ```bash
   make setup
   ```

2. **Prepare the dataset**
   ```bash
   make data
   ```
   When PCam assets are not present the command falls back to generating a small synthetic dataset that mimics the structure of the real data.  This behaviour keeps the repository fully testable while still reflecting the real execution flow.

3. **Train baseline models**
   ```bash
   make train_stylegan
   make train_diffusion
   ```
   The default implementations create light-weight mock checkpoints so that downstream steps can run deterministically without requiring GPU resources.  Replace the training hooks in `src/models/` with production grade trainers to obtain research quality results.

4. **Generate samples, evaluate, and create a report**
   ```bash
   make sample
   make eval
   make report
   ```

5. **Launch the interactive demo**
   ```bash
   make demo
   ```

6. **Full reproducibility**
   ```bash
   make reproduce
   ```
   This command chains all previous steps to rebuild the complete pipeline from scratch.

## Testing

Run the unit test suite with:

```bash
pytest
```

The tests validate image IO helpers, evaluation metrics, and basic privacy analytics to guarantee a stable interface for extending the project.

## License

This project is distributed under the terms of the MIT license.  See [LICENSE](LICENSE).
