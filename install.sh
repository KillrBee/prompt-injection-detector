#!/bin/bash
# Installation script for Prompt Injection Detector on macOS
# Handles common Apple Silicon installation issues

set -e  # Exit on error

echo "================================================"
echo "Prompt Injection Detector - Installation"
echo "macOS (Apple Silicon) Version"
echo "================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: macOS required${NC}"
    exit 1
fi

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_CMD="python3"

# Try to find Python 3.10 or 3.11 specifically
for py in python3.11 python3.10 python3; do
    if command -v $py &> /dev/null; then
        VERSION=$($py --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [[ $MAJOR -eq 3 ]] && [[ $MINOR -ge 10 ]] && [[ $MINOR -le 11 ]]; then
            PYTHON_CMD=$py
            break
        fi
    fi
done

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Using $PYTHON_CMD (version $PYTHON_VERSION)${NC}"

# Check/Install Homebrew
echo -e "\n${YELLOW}Checking Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add to PATH for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}✓ Homebrew found${NC}"
fi

# Install system dependencies
echo -e "\n${YELLOW}Installing system dependencies...${NC}"
brew install libomp || echo "libomp already installed"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Removing existing venv..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install PyTorch for Apple Silicon
echo -e "\n${YELLOW}Installing PyTorch with Apple Silicon support...${NC}"
pip install torch torchvision torchaudio

# Install requirements in stages to handle failures gracefully
echo -e "\n${YELLOW}Installing core dependencies...${NC}"

# Stage 1: Core scientific stack
pip install numpy pandas scikit-learn scipy

# Stage 2: Gradient boosting (can fail on some systems)
echo -e "\n${YELLOW}Installing XGBoost and LightGBM...${NC}"
pip install xgboost || echo "${YELLOW}Warning: XGBoost install failed, will retry${NC}"
pip install lightgbm || echo "${YELLOW}Warning: LightGBM install failed, will retry${NC}"

# Stage 3: Transformers and NLP
echo -e "\n${YELLOW}Installing transformers and NLP libraries...${NC}"
pip install transformers sentence-transformers
pip install nltk spacy

# Stage 4: Text analysis
pip install py-readability-metrics textstat pyspellchecker

# Stage 5: Utilities
pip install tqdm joblib python-dotenv pyyaml

# Stage 6: API/Service
pip install flask flask-cors gunicorn

# Stage 7: Model export
pip install onnx skl2onnx onnxruntime

# Stage 8: Data handling
pip install datasets huggingface-hub

# Verify critical packages
echo -e "\n${YELLOW}Verifying installations...${NC}"
MISSING=()

python3 << 'EOFPY'
import sys
packages = [
    'numpy', 'pandas', 'sklearn', 'scipy',
    'torch', 'transformers', 'sentence_transformers',
    'nltk', 'spacy', 'textstat',
    'flask', 'joblib', 'yaml'
]

missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")
        missing.append(pkg)
        
if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
EOFPY

if [ $? -ne 0 ]; then
    echo -e "${RED}Some packages failed to install. Trying pip install -r requirements.txt...${NC}"
    pip install -r requirements.txt || echo "${YELLOW}Some optional packages skipped${NC}"
fi

# Download NLTK data
echo -e "\n${YELLOW}Downloading NLTK data...${NC}"
python3 << 'EOFPY'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
for package in ['punkt', 'punkt_tab', 'stopwords', 'vader_lexicon', 
                'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(package, quiet=True)
        print(f"✓ {package}")
    except Exception as e:
        print(f"✗ {package}: {e}")

print("✓ NLTK data download complete")
EOFPY

# Download spaCy model
echo -e "\n${YELLOW}Downloading spaCy model...${NC}"
python3 -m spacy download en_core_web_sm --quiet || \
    python3 -m spacy download en_core_web_sm

# Create directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs
mkdir -p exports

# Download and cache transformer models
echo -e "\n${YELLOW}Caching transformer models (this may take a few minutes)...${NC}"
python3 << 'EOFPY'
from sentence_transformers import SentenceTransformer
import sys

models_to_cache = [
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-MiniLM-L6-v2',
]

print("\nDownloading sentence transformer models...")
for model_name in models_to_cache:
    try:
        print(f"  - {model_name}...", end=' ', flush=True)
        model = SentenceTransformer(model_name)
        dims = model.get_sentence_embedding_dimension()
        print(f"✓ ({dims} dims)")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

print("\n✓ All models cached successfully")
EOFPY

# Create .env file
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cat > .env << 'EOFENV'
# Prompt Injection Detector Configuration

# Model settings
EMBEDDING_MODEL=all-mpnet-base-v2
BERT_MODEL=ProtectAI/deberta-v3-base-prompt-injection-v2

# Training settings
BATCH_SIZE=32
LEARNING_RATE=0.05
N_ESTIMATORS=200
MAX_DEPTH=6

# API settings
FLASK_PORT=5555
FLASK_DEBUG=False

# Paths
DATA_DIR=./data
MODEL_DIR=./data/models
LOG_DIR=./logs
EOFENV
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\n${YELLOW}To activate the environment:${NC}"
echo -e "  source venv/bin/activate"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Download datasets: ${GREEN}python scripts/download_datasets.py${NC}"
echo -e "  2. Train model:       ${GREEN}python scripts/train.py${NC}"
echo -e "  3. Test detector:     ${GREEN}python scripts/test_inference.py${NC}"
echo -e "\n${YELLOW}Verify installation:${NC}"
echo -e "  python scripts/verify_install.py"
