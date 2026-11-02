# README

Code for **See First, Answer Later: Visual Evidence Pre-Alignment via Sufficiency-Driven RL**.

VEPA trains multimodal large language models to generate question-conditioned visual evidence through GRPO. A frozen text-only reader provides answer-recovery feedback, and the aligned model initializes downstream supervised fine-tuning.

## Environment

The implementation uses [VeRL](https://github.com/verl-project/verl), with PyTorch FSDP for training and vLLM for generation.

Prepare a Linux GPU environment following the [VeRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html). Required packages include PyTorch, vLLM, Ray, Transformers, Hydra, OmegaConf, NumPy, and Pillow. The default logging configuration also uses TensorBoard and SwanLab.

The released adapter uses VeRL's `RayPPOTrainer` and `RLHFDataset` interfaces. Dependency versions should match your VeRL checkout.

## Preparation

### Clone the repository

Run the following from the root of your configured VeRL checkout:

```bash
git clone https://github.com/Winnie-Lian/VisualEvidencePreAlignment.git \
  examples/VisualEvidencePreAlignment

cd examples/VisualEvidencePreAlignment
```

Keep this directory structure, as the training script locates VeRL's base configuration through a relative path.

### Prepare data and models

Prepare training and validation Parquet files with the following fields:

| Field | Content |
| --- | --- |
| `prompt` | Question text |
| `images` | Image path |
| `response` | Reference answer |

Include any additional metadata required by your VeRL data loader.

The default policy is **Qwen2-VL-2B**, and the frozen reader is **Qwen2.5-7B-Instruct**. Update these paths before training:

| File | Settings |
| --- | --- |
| `config.yaml` | `data.train_files`, `data.val_files`, `actor_rollout_ref.model.path` |
| `train_caption_grpo.py` | `AUX_MODEL_PATH` and `OUTPUT_DIR` inside `main()` |

Training data and model weights are not included in this repository.

## Training

The supplied configuration allocates **7 GPUs to the policy/reference pool and 1 GPU to the frozen reader**.

Adjust training settings in `config.yaml`, then launch:

```bash
python train_caption_grpo.py
```

Configurations, rollout logs, and training checkpoints are stored under:

```text
<OUTPUT_DIR>/results/
```

### Code structure

```text
├── train_caption_grpo.py   # Dataset adapter, prompts, and GRPO training
├── auxiliary_worker.py     # Frozen reader and reward computation
└── config.yaml             # Training configuration
```

## Acknowledgements

Built on [VeRL](https://github.com/verl-project/verl), [vLLM](https://github.com/vllm-project/vllm), and [Qwen](https://github.com/QwenLM).
