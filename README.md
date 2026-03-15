# 🧬 CareerDNA — Neural Career Intelligence Platform

> **Real AI. Real Flask. Zero hardcoding.**
> Every result comes from the Python + TensorFlow backend.
> Opening index.html directly without app.py will show "Backend Offline" — that is correct behaviour.

---

## ⚠️ Important — This Is A Backend-Required Project

| Situation | Result |
|---|---|
| Open `index.html` directly (no Flask) | ❌ Shows "Backend Offline" error |
| Run `python app.py` then open browser | ✅ All 5 features work with real AI |

---

## 🚀 What CareerDNA Does

Five AI-powered career tools in one web application:

| # | Feature | What It Does | AI Method |
|---|---|---|---|
| 1 | 🧬 Career Path Predictor | Ranks 14 career paths for your skill set | TensorFlow ANN + Softmax |
| 2 | 📉 Skill Decay Detector | Rates each skill as Rising/Stable/Declining | Velocity scoring + ReLU normalisation |
| 3 | 🎯 Interview IQ Scorer | Scores readiness topic-by-topic from any JD | TF-IDF + Cosine Similarity |
| 4 | 📄 ATS Resume Checker | 10-point ATS analysis like Taleo/Workday | Weighted keyword matching engine |
| 5 | 🗺️ Learning Roadmap | Week-by-week personalised learning plan | Cosine similarity gap analysis |

Every feature also recommends **free YouTube videos + courses** based on your skill gaps.

---

## 📁 Project Structure

```
careerdna/
│
├── app.py                  ← Flask backend — ALL AI logic lives here
├── train_model.py          ← Trains the TensorFlow ANN (run first, once)
├── requirements.txt        ← Python package list
├── README.md               ← This setup guide
├── DOCUMENTATION.md        ← Full NNDL technical documentation
│
└── templates/
    └── index.html          ← Frontend UI — only calls Flask API, no fake JS
```

After training, a `model/` folder is auto-created:
```
model/
├── career_model.keras      ← Trained ANN weights (~180 KB)
├── scaler.pkl              ← MinMaxScaler (fitted on training data)
├── training_history.json   ← Loss and accuracy per epoch
└── metadata.json           ← Architecture info, accuracy, skill index
```

---

## ⚡ Setup Guide (5 Steps)

### Step 1 — Check Python version
```bash
python --version
```
You need **Python 3.10 or higher**.
Download from: https://www.python.org/downloads/

---

### Step 2 — Open terminal inside the careerdna folder

**Windows:**
- Right-click the `careerdna` folder → "Open in Terminal"
- Or: Win+R → type `cmd` → `cd C:\path\to\careerdna`

**Mac/Linux:**
```bash
cd /path/to/careerdna
```

---

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

This installs: Flask, TensorFlow, NumPy, Pandas, scikit-learn.

> ⏱️ First install takes 3–5 minutes (TensorFlow is ~500 MB).
> If `pip` not found, try `pip3` or `python -m pip`

---

### Step 4 — Train the ANN model (first time only)
```bash
python train_model.py
```

**What happens:**
- Generates 2,000 synthetic career training samples
- Trains a 4-layer ANN with Softmax output
- Saves model to `model/` folder

**Expected output:**
```
============================================================
  CareerDNA — ANN Training
============================================================
  Careers:   14
  Features:  30 skills
  Dataset:   2000 samples
  Normalised: MinMaxScaler → [0, 1]
  Split:     Train=1400 | Val=300 | Test=300

  Architecture:
    Input(30) → Dense(128)+BN+ReLU+Drop(0.3)
                         → Dense(64)+BN+ReLU+Drop(0.2)
                         → Dense(32)+ReLU
                         → Output(14, Softmax)
    Parameters: 15,534

  Training (max 150 epochs, early stopping)...
  ...
  Test Accuracy: 94.3%

  Saved: model/career_model.keras
  Done! Now run: python app.py
============================================================
```

> You only need to run this **once**. The saved model is reloaded every time you start app.py.

---

### Step 5 — Start the Flask server
```bash
python app.py
```

You will see:
```
==================================================
  CareerDNA — Neural Career Intelligence
  http://localhost:5000
==================================================
[OK] ANN loaded — 15,534 params, 94.3% accuracy
 * Running on http://127.0.0.1:5000
```

Open your browser and go to:
```
http://localhost:5000
```

**The green status bar at the top means AI is live. ✅**

---

## 🔌 API Routes

All analysis happens in `app.py`. The frontend sends POST requests to these routes:

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serves the web app |
| GET | `/api/status` | Returns server + model status |
| POST | `/api/career` | Career path prediction (TF ANN) |
| POST | `/api/decay` | Skill decay analysis |
| POST | `/api/interview` | Interview IQ scoring (TF-IDF) |
| POST | `/api/ats` | ATS resume analysis (10-point) |
| POST | `/api/roadmap` | Learning roadmap generation |

### Example — Test API directly
```bash
# While app.py is running, open another terminal:
curl -X POST http://localhost:5000/api/career \
  -H "Content-Type: application/json" \
  -d '{"skills":["python","tensorflow","docker"],"exp_level":0,"work_style":"builder","interests":["ai/ml"]}'
```

---

## 🧠 NNDL Concepts Implemented (12 Total)

| # | Concept | File | Function |
|---|---|---|---|
| 1 | Multi-Class ANN | train_model.py | build_model() |
| 2 | Softmax Activation | train_model.py | Output layer |
| 3 | ReLU Activation | train_model.py | All hidden layers |
| 4 | Dropout (0.3 + 0.2) | train_model.py | Regularisation |
| 5 | Batch Normalisation | train_model.py | After each Dense |
| 6 | Adam Optimizer | train_model.py | compile() |
| 7 | Cross-Entropy Loss | train_model.py | compile() |
| 8 | MinMax Normalisation | train_model.py | MinMaxScaler |
| 9 | Cosine Similarity | app.py | cosine_similarity() |
| 10 | TF-IDF | app.py | tfidf_weight() |
| 11 | Backpropagation | train_model.py | model.fit() |
| 12 | Early Stopping | train_model.py | callbacks |

See `DOCUMENTATION.md` for full explanation of each concept with code.

---

## 🔧 Troubleshooting

**`pip install` fails:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Port 5000 already in use:**
```python
# Edit last line of app.py:
app.run(debug=True, port=5001)
# Then open: http://localhost:5001
```

**"Model not found" error when starting app.py:**
```bash
python train_model.py   # must run this first!
```

**TensorFlow install fails on Mac M1/M2:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal   # optional GPU acceleration
```

**`python` command not found on Windows:**
- Use `py` instead: `py train_model.py`, `py app.py`

---

## 📦 Dependencies

```
flask==3.0.3          Web server framework
tensorflow==2.16.1    Neural network training + inference
numpy==1.26.4         Array operations
pandas==2.2.2         Data handling
scikit-learn==1.5.0   MinMaxScaler, train_test_split
```

---

## 💻 System Requirements

| Item | Minimum |
|---|---|
| Python | 3.10 or higher |
| RAM | 4 GB (8 GB recommended for training) |
| Disk | 2 GB (for TensorFlow) |
| OS | Windows 10/11, macOS 11+, Ubuntu 20.04+ |
| Browser | Chrome, Firefox, Edge (any modern browser) |

---

*CareerDNA — NNDL Course Project 🧬*
