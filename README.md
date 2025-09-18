# 🎮 FrozenLake Game Agent

A reinforcement learning (RL) agent trained on OpenAI’s **FrozenLake-v1** environment.  
The project started as a simple Q-learning exercise and was gradually built up to follow **industry-style MLOps practices**.

---

## What this project is about

- **Core Goal**: Train a Q-learning agent to solve FrozenLake (8x8 map, deterministic).  
- **Why it matters**: Shows how a basic RL example can be treated like a real ML product, with testing, reproducibility, and deployment in mind.  
- **Personal Aim**: Learn and demonstrate MLOps best practices step by step, not just “make it run.”

---

## Features Implemented

### Level 1 — Core RL Agent
- Implemented **Q-learning** with epsilon-greedy exploration.  
- Added reproducibility options (seeds).  
- Saved artifacts (metrics, trained Q-tables).  
- Configurable environment (`configs/frozenlake.yaml`).  

### Level 2 — Testing & CI/CD
- ✅ Unit tests with **pytest**.  
- ✅ Continuous integration via **GitHub Actions** (runs tests on every push).  
- ✅ Stable test design (trivial Q-table checks for consistency).  
- ✅ Coverage reporting (`pytest-cov`).  

### Level 3 — Productionization
- ✅ **Dockerfile** + `.dockerignore` → reproducible builds.  
- ✅ CI pipeline extended to **build & run Docker image**.  
- ✅ **Linting checks** with `flake8`.  
- ✅ Configs in YAML (separation of code vs parameters).  

### Level 4 — API & Deployment (in progress)
- ✅ Added a **FastAPI service** (`fastapi_app.py`) to expose the agent via an API.  
- Example endpoints:  
  - `/` → health check  
  - `/act?state=3` → get action for state=3 from the trained Q-table  
- API runs via **Uvicorn**, can be containerized with Docker.  

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/charlesl3/frozenlake-game-agent.git
cd frozenlake-game-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run training locally
```bash
python main.py
```

### 4. Run tests
```bash
pytest -q
```

### 5. Run with coverage
```bash
pytest --cov=. --cov-report=term-missing -q
```

### 6. Run FastAPI (local server)
```bash
uvicorn fastapi_app:app --reload
```
Then open in browser:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Example: [http://127.0.0.1:8000/act?state=3](http://127.0.0.1:8000/act?state=3)  

### 7. Docker (training + API inside container)
```bash
docker build -t frozenlake-agent .
docker run --rm -p 8000:8000 frozenlake-agent
```

---

## 📂 Repo Structure
```
FrozenLake/
├── main.py              # Entry point
├── qlearning.py         # Training & evaluation logic
├── policies.py          # Agent’s action policies
├── utils/               # Logging, helpers
├── configs/             # YAML configs
│   └── frozenlake.yaml
├── test/                # Pytest unit tests
│   └── test_qlearning.py
├── artifacts/           # Saved metrics & Q-table
├── logs/                # Training logs
├── fastapi_app.py       # FastAPI service (expose trained agent)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml   # CI/CD pipeline
```

---

## Why this project is useful
- Demonstrates **RL concepts** in a very clear environment.  
- Doubles as a **mini MLOps pipeline**: CI/CD, Docker, linting, coverage, API.  
- Strong **resume project** — signals readiness for DS/DE/Quant roles.  

---

## Next Steps (future polish)
- Add CI and coverage badges to README.  
- Deploy FastAPI app to Render/Heroku for public demo.  
- (Optional) Add visualization (training curves, Q-table heatmap).  
