"""Verify installation"""
import sys

def check_imports():
    errors = []
    
    modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('nltk', 'NLTK'),
        ('spacy', 'spaCy'),
        ('textstat', 'Textstat'),
    ]
    
    print("Checking imports...")
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            errors.append(name)
    
    # Check spaCy model
    try:
        import spacy
        spacy.load('en_core_web_sm')
        print("✓ spaCy English model")
    except:
        print("✗ spaCy English model not found")
        errors.append("spaCy model")
    
    # Check NLTK data
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK data")
    except:
        print("✗ NLTK data not found")
        errors.append("NLTK data")
    
    if errors:
        print(f"\n❌ Installation incomplete. Missing: {', '.join(errors)}")
        return 1
    else:
        print("\n✅ Installation verified successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(check_imports())
