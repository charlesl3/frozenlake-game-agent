# 🎮 FrozenLake Game Agent

A reinforcement learning (RL) agent trained on OpenAI’s **FrozenLake-v1** environment.  
The project started as a simple Q-learning exercise and was gradually built up to follow **industry-style MLOps practices**.

---

## ✨ What this project is about

- **Core Goal**: Train a Q-learning agent to solve FrozenLake (8x8 map, deterministic).  
- **Why it matters**: Shows how a basic RL example can be treated like a real ML product, with testing, reproducibility, and deployment in mind.  
- **Personal Aim**: Learn and demonstrate MLOps best practices step by step, not just “make it run.”

---

## 🛠 Features Implemented

### Level 1 — Core RL agent
- Implemented **Q-learning** with epsilon-greedy exploration.  
- Added reproducibility options (seeds).  
- Saved artifacts (metrics, trained Q-tables).  
- Configurable environment (`configs/frozenlake.yaml`).

### Level 2 — Testing & CI/CD
- ✅ Unit tests with **pytest**.  
- ✅ Continuous integration via **GitHub Actions** (runs tests on every push).  
- ✅ Stable test design (trivial Q-table checks for consistency).  

### Level 3 — Productionization
- ✅ **Dockerfile** + `.dockerignore` → reproducible environment.  
- ✅ CI pipeline extended to **build and run Docker image**.  
- ✅ **Coverage reports** with `pytest-cov` integrated into CI.  
- ✅ **Linting checks** with `flake8`.  
- ✅ Configs in YAML (separation of code vs parameters).  
- ✅ (Coming next) CI badge in README.  

---

## 🚀 How to Run

### Local (Python only)
```bash
pip install -r requirements.txt
python main.py
```

### Run tests
```bash
pytest -q
```

### With coverage
```bash
pytest --cov=. --cov-report=term-missing -q
```

### Docker
```bash
docker build -t frozenlake-agent .
docker run --rm frozenlake-agent
```

---

## 📂 Repo Structure
```
FrozenLake/
├── main.py              # Entry point
├── qlearning.py         # Training & evaluation logic
├── config.py            # Config loader (with YAML)
├── configs/             # YAML configs
│   └── frozenlake.yaml
├── test/                # Pytest unit tests
├── artifacts/           # Saved metrics & models
├── logs/                # Training logs
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml   # CI/CD pipeline
```

---

## 🔍 Why this project is useful
- Demonstrates **RL concepts** in a very clear environment.  
- Doubles as a **mini MLOps pipeline**: CI/CD, Docker, linting, coverage.  
- Strong **resume project** — signals readiness for DS/DE/Quant roles.

---

## 🏆 Next Steps (for myself)
- Add CI badge to README.  
- (Optional) Coverage badge via Codecov.  
- (Optional) Wrap as a FastAPI service (`/act?state=3`).  
