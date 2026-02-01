# Installation Guide

Detailed installation instructions for macOS.

## Prerequisites

### 1. Check Python Version

```bash
python3 --version
```

You need Python 3.10 or 3.11. If you don't have it:

```bash
# Install via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

### 2. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Install System Dependencies

```bash
# Required for LightGBM on macOS
brew install libomp

# Required for some Python packages
brew install gcc
```

## Installation Steps

### Step 1: Clone or Download Project

If using git:
```bash
git clone https://github.com/yourusername/prompt-injection-detector.git
cd prompt-injection-detector
```

Or download and extract the zip file, then:
```bash
cd prompt-injection-detector
```

### Step 2: Run Installation Script

The installation script will:
- Create a virtual environment
- Install all Python dependencies
- Download NLTK data
- Download spaCy models
- Cache transformer models

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

This will take 10-20 minutes depending on your internet connection.

### Step 3: Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Verify Installation

```bash
python scripts/verify_install.py
```

This will check that all dependencies are correctly installed.

## Manual Installation (if automated script fails)

If the installation script fails, you can install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (Apple Silicon optimized)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Common Issues

### Issue: "libomp not found"

**Solution:**
```bash
brew install libomp
```

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: PyTorch not using Apple Silicon GPU

**Solution:** PyTorch will automatically use MPS (Metal Performance Shaders) on Apple Silicon. You can verify with:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Issue: NLTK data download fails with SSL error

**Solution:**
```bash
pip install --upgrade certifi
```

Then run the download again.

### Issue: Out of memory during training

**Solution:** Reduce batch size in `config/config.yaml`:

```yaml
training:
  batch_size: 16  # Reduce from 32
```

## Next Steps

After successful installation:

1. **Download datasets**: `python scripts/download_datasets.py`
2. **Train model**: `python scripts/train.py`
3. **Export model**: `python scripts/export_onnx.py`
4. **Test inference**: `python scripts/test_inference.py`

## Uninstallation

To remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf prompt-injection-detector
```

## Support

For issues, please check the troubleshooting section or open an issue on GitHub.
