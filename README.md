# CT-Data: Vision-Language Preprocessing and Training for Chest CT

This repository contains code for building Vision-Language Preprocessing (VLP) and Vision-Language Models (VLM) for Chest CT scans using the CT-RATE dataset. The project includes comprehensive preprocessing pipelines, training scripts, and evaluation tools for medical imaging applications.

## 🏥 Project Overview

This codebase is designed to create VLP and VLM models specifically for Chest CT imaging using the CT-RATE dataset. It provides:

- **Preprocessing Pipeline**: Convert raw CT scans (.nii.gz) to optimized formats for training
- **Training Framework**: Multi-GPU distributed training with DeepSpeed integration
- **Vision-Language Models**: Integration with LLM2Vec and custom CLIP architectures
- **Evaluation Tools**: Comprehensive evaluation metrics for medical imaging tasks

## 🚀 Quick Start

### Prerequisites

- **Docker**: Latest version recommended
- **NVIDIA GPU**: 6+ GPUs with CUDA support
- **Data**: CT-RATE dataset access

### 1. Installation

Start the Docker container using Docker Compose:

```bash
docker compose up -d
```

**Important Notes:**
- Ensure you have the latest Docker version installed
- Check container ID/name to avoid conflicts
- Verify volume mounts are correctly configured
- The container is configured for 6 NVIDIA GPUs with 64GB shared memory

#### Install LLM2Vec4CXR

After starting the Docker container, install the LLM2Vec4CXR package:

```bash
git clone https://github.com/lukeingawesome/llm2vec4cxr.git
cd llm2vec4cxr
pip install -e .
```

This installs the LLM2Vec4CXR package in development mode, which provides the LLM2Vec models for chest X-ray report analysis used in this project.

### 2. Data Preprocessing

The preprocessing pipeline is implemented in `preprocess.py`:

```bash
python preprocess.py --src /path/to/raw/ct/scans --dst /path/to/processed/data
```

**Preprocessing Features:**
- Converts .nii.gz files to optimized .npz format
- Standardizes spacing to (1.25, 1.25, 2.0) mm
- Resizes to (256, 256, 192) voxels
- Applies HU windowing (-1000, 1500)
- Multi-processed for efficiency

### 3. Training

Execute training using the provided shell script:

```bash
./run.sh
```

**Training Configuration:**
- **Model**: LLM2Vec + CLIP architecture
- **GPUs**: 6 GPUs with DeepSpeed ZeRO Stage 1
- **Batch Size**: 12 per GPU
- **Precision**: BF16 mixed precision
- **Optimizer**: AP-AdamW with gradient clipping
- **Monitoring**: TensorBoard and Weights & Biases integration

### 4. Evaluation

Evaluation code is currently being implemented and will be available soon.

## 📁 Project Structure

```
CT-Data/
├── preprocess.py          # CT scan preprocessing pipeline
├── run.sh                 # Training execution script
├── docker-compose.yml     # Docker container configuration
├── Dockerfile            # Container image definition
├── requirements.txt      # Python dependencies
├── training/             # Training framework
│   ├── main.py          # Main training script
│   ├── data.py          # Data loading and augmentation
│   ├── model_utils.py   # Model utilities
│   └── ...              # Additional training modules
├── csv/                  # Dataset CSV files
├── data.ipynb           # Data exploration notebook
├── visualize.ipynb      # Visualization notebook
└── eval_finetune.py     # Evaluation script (in development)
```

## 🔧 Configuration

### Docker Configuration

The `docker-compose.yml` includes:
- NVIDIA GPU support (6 GPUs)
- Volume mounts for data and home directory
- 64GB shared memory for 3D workloads
- Interactive shell for development

### Training Parameters

Key training parameters in `run.sh`:
- **Learning Rate**: 1e-4 (visual, text, projection)
- **Weight Decay**: 0.05
- **Epochs**: 30
- **Warmup Steps**: 1000
- **Gradient Accumulation**: 1 step
- **Mixed Precision**: BF16

## 📊 Data Format

### Input Data
- **Raw CT Scans**: .nii.gz files with original spacing and dimensions
- **CSV Metadata**: Contains image paths, findings, and split information

### Processed Data
- **NPZ Files**: Compressed numpy arrays with standardized format
- **Metadata CSV**: Processing statistics and file mappings

## 🧠 Model Architecture

The project implements:
- **Vision Encoder**: Custom CLIP-based architecture for CT scans
- **Text Encoder**: LLM2Vec with medical domain adaptation
- **Projection Layer**: Learned alignment between vision and text features
- **Loss Functions**: Contrastive learning with medical-specific augmentations

## 📈 Monitoring

Training progress is monitored through:
- **TensorBoard**: Local training metrics
- **Weights & Biases**: Cloud-based experiment tracking
- **Logging**: Comprehensive training logs

## 🔬 Research Applications

This codebase supports:
- Medical image-text retrieval
- Radiology report generation
- Multi-modal medical AI
- Clinical decision support systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

?

## 🙏 Acknowledgments

- CT-RATE dataset providers
- MONAI framework for medical imaging
- DeepSpeed for distributed training
- LLM2Vec for text encoding

## 📞 Contact

lucasko1994@snu.ac.kr

---

**Note**: This project is actively under development. Evaluation code and additional features will be added in future updates.
