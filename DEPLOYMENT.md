# OpenClaw Integration Guide

How to integrate the prompt injection detector with OpenClaw.

## Architecture

```
OpenClaw Message Flow:
User → Channel → OpenClaw Gateway → Hook (Detector) → Agent → Response
                                      ↓
                              ML Classifier (50ms)
                              ↓
                        Block/Allow/Flag
```

## Option 1: Python Microservice (Recommended)

### Setup

1. **Install detector in VM**:
```bash
cd ~/prompt-injection-detector
./install.sh
source venv/bin/activate
python scripts/download_datasets.py
python scripts/train.py
```

2. **Create Flask API** (`api_server.py`):
```python
from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# Load models
model_dir = Path('./data/models')
extractor = joblib.load(model_dir / 'feature_extractor_latest.joblib')
xgb = joblib.load(model_dir / 'xgboost_latest.joblib')
lgb = joblib.load(model_dir / 'lightgbm_latest.joblib')

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json
    text = data.get('text', '')
    
    # Extract features
    features_dict = extractor.extract(text)
    import pandas as pd
    X = pd.DataFrame([features_dict])
    
    # Predict
    xgb_prob = xgb.predict_proba(X)[0, 1]
    lgb_prob = lgb.predict_proba(X)[0, 1]
    prob = (xgb_prob + lgb_prob) / 2
    
    return jsonify({
        'is_injection': bool(prob > 0.5),
        'confidence': float(prob),
        'threshold': 0.5
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555)
```

3. **Run API server**:
```bash
python api_server.py
```

### OpenClaw Hook Integration

Create `~/.openclaw/hooks/ml-injection-scanner.js`:

```javascript
// ML-based prompt injection scanner
const axios = require('axios');

const SCANNER_URL = 'http://127.0.0.1:5555/scan';
const CONFIDENCE_THRESHOLD = 0.8;

module.exports = {
  name: 'ml-injection-scanner',
  version: '1.0.0',
  
  async beforeAgentMessage(context) {
    const { message, agent } = context;
    
    // Skip for trusted sources
    if (message.channel === 'cli') {
      return;
    }
    
    try {
      const response = await axios.post(SCANNER_URL, {
        text: message.content
      }, {
        timeout: 2000  // 2s timeout
      });
      
      const { is_injection, confidence } = response.data;
      
      if (is_injection && confidence > CONFIDENCE_THRESHOLD) {
        console.warn(`🚨 Blocked injection (${(confidence*100).toFixed(1)}%): ${message.content.substring(0, 50)}...`);
        
        throw new Error(
          `This message was blocked by security filters. ` +
          `If you believe this is an error, please rephrase your request.`
        );
      }
      
      // Log suspicious but not blocked
      if (confidence > 0.3 && confidence <= CONFIDENCE_THRESHOLD) {
        console.warn(`⚠️  Suspicious message (${(confidence*100).toFixed(1)}%): ${message.content.substring(0, 50)}...`);
      }
      
    } catch (error) {
      if (error.message.includes('blocked by security')) {
        throw error;  // Re-throw blocking errors
      }
      
      // On API failure, log but don't block
      console.error('ML scanner error:', error.message);
    }
  }
};
```

### Enable Hook

In `~/.openclaw/openclaw.json`:

```json
{
  "hooks": {
    "enabled": true,
    "directory": "/Users/yourusername/.openclaw/hooks"
  }
}
```

Restart OpenClaw:
```bash
openclaw restart
```

## Option 2: ONNX Runtime (Faster)

For production deployment, export to ONNX for 2-3x faster inference:

```bash
python scripts/export_onnx.py
npm install onnxruntime-node
```

Update hook to use ONNX:

```javascript
const ort = require('onnxruntime-node');

// Load models once
let xgbSession, lgbSession, extractor;

async function initModels() {
  xgbSession = await ort.InferenceSession.create('./exports/xgboost_model.onnx');
  lgbSession = await ort.InferenceSession.create('./exports/lightgbm_model.onnx');
  extractor = require('./feature_extractor.js');  // Ported to JS
}

async function scanText(text) {
  const features = extractor.extract(text);
  const tensor = new ort.Tensor('float32', features, [1, features.length]);
  
  const xgbResult = await xgbSession.run({ float_input: tensor });
  const lgbResult = await lgbSession.run({ float_input: tensor });
  
  const xgbProb = xgbResult.probabilities.data[1];
  const lgbProb = lgbResult.probabilities.data[1];
  
  return (xgbProb + lgbProb) / 2;
}
```

## Performance Tuning

### Reduce Latency

1. **Use smaller embedding model**:
   - Change `all-mpnet-base-v2` → `all-MiniLM-L6-v2` in `config/config.yaml`
   - Reduces from 768 → 384 dimensions
   - 2x faster, ~3% accuracy drop

2. **Disable BERT features**:
   - Set `use_bert_features: false` in config
   - Saves ~50ms per request

3. **Batch processing**:
   - For email/document scanning, batch multiple items

### Memory Optimization

If running out of memory in VM:

```yaml
# config/config.yaml
training:
  batch_size: 16  # Reduce from 32
```

## Monitoring

Add logging to track detections:

```javascript
const fs = require('fs');

function logDetection(message, confidence, blocked) {
  const entry = {
    timestamp: new Date().toISOString(),
    message: message.substring(0, 100),
    confidence,
    blocked
  };
  
  fs.appendFileSync(
    '/var/log/openclaw/injections.jsonl',
    JSON.stringify(entry) + '\n'
  );
}
```

## Security Considerations

1. **Defense in Depth**: ML detector is first layer, not only layer
2. **Tunable Threshold**: Adjust `CONFIDENCE_THRESHOLD` based on false positive rate
3. **Allowlist**: Skip scanning for known-safe sources
4. **Rate Limiting**: Prevent API DoS

## Testing

Test the integration:

```bash
# Terminal 1: Start API
cd ~/prompt-injection-detector
source venv/bin/activate
python api_server.py

# Terminal 2: Test OpenClaw
openclaw chat --agent main
# Try: "Ignore all previous instructions"
# Should be blocked

# Try: "What is the weather?"
# Should work normally
```

## Troubleshooting

### Hook not triggering
- Check `openclaw.json` hooks config
- Verify hook file has correct exports
- Check logs: `~/.openclaw/logs/`

### API connection errors
- Verify API running: `curl http://127.0.0.1:5555/health`
- Check firewall rules
- Increase timeout in hook

### High false positive rate
- Increase `CONFIDENCE_THRESHOLD` to 0.85-0.9
- Retrain with more benign examples
- Add domain-specific examples to training data

## Performance Benchmarks

| Configuration | Latency | Accuracy |
|--------------|---------|----------|
| Full (mpnet + BERT) | ~150ms | 85-87% |
| Standard (mpnet) | ~80ms | 82-85% |
| Fast (MiniLM) | ~50ms | 78-82% |
| ONNX (MiniLM) | ~30ms | 78-82% |

Choose based on your requirements.
