# SIAO-CNN-ORNN

**Self-Improved Aquila Optimizer enhanced CNN-ORNN for Nuclear Reactor Fault Detection**

A hybrid deep learning framework for monitoring faults and analyzing reliability in the IP-200 small modular reactor safety-critical systems.

---

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   1. Data       │───▶│  2. Features    │───▶│   3. Models     │
│  Acquisition &  │    │   Extraction    │    │   CNN-ORNN      │
│  Preprocessing  │    │     (WKS)       │    │                 │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐             │
│  5. Inference   │◀───│  4. Optimizers  │◀────────────┘
│  Reliability    │    │     SIAO        │
└─────────────────┘    └─────────────────┘
```

---

##  Project Structure

```
SIAO-CNN-ORNN-Deployment/
│
├── stage1_data/             # Stage 1: Data Acquisition & Preprocessing
├── stage2_features/         # Stage 2: Feature Extraction (WKS)
├── stage3_models/           # Stage 3: CNN-ORNN Architecture
├── stage4_optimizers/       # Stage 4: SIAO Weight Optimization
├── stage5_inference/        # Stage 5: Prediction & Reliability
│
├── data/                    # Dataset (organized by fault type)
│   ├── steady_state/
│   ├── transient/
│   ├── porv/
│   ├── sgtr/
│   ├── fwlb/
│   └── rcp_failure/
│
├── notebooks/               # Jupyter notebooks
│   └── model_logic.ipynb    # Training notebook (for Colab)
│
├── train_pipeline.py        # Main training script
├── pyproject.toml           # UV package configuration
└── README.md
```

---

##  Quick Start

### Local Setup (UV)

```bash
# Install UV if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -e .
```

### Google Colab

Open `notebooks/model_logic.ipynb` in Colab. The setup cell will install dependencies:

```python
!pip install torch numpy pandas scipy scikit-learn matplotlib openpyxl

import sys
sys.path.insert(0, '/content/SIAO-CNN-ORNN-Deployment')
```

---

##  Fault Classes

| Class | Fault Type | Severity |
|:-----:|------------|----------|
| 0 | Steady State | Normal |
| 1 | Transient (Power Change) | Minor |
| 2 | PORV Stuck Open | Significant |
| 3 | SGTR (Steam Generator Tube Rupture) | Critical |
| 4 | FWLB (Feedwater Line Break) | Critical |
| 5 | RCP Failure (Reactor Coolant Pump) | Severe |

---

##  Key Features

- **WKS Feature Extraction**: Weighted Kurtosis-Skewness with Aquila-optimized weights
- **SIAO Optimization**: Chaotic map-enhanced Aquila Optimizer for RNN weights
- **Real-time Inference**: Sliding window buffer for streaming sensor data
- **Reliability Analysis**: MTTF, failure rate, and maintenance planning

---

## License

MIT License
