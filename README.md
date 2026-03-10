# Tech-App-Devoir-II

**INF 6243 — Classification et Apprentissage Automatique**

Assignment project for machine learning / classification. This repo is set up for local development with a Python virtual environment and optional CUDA support on Linux and Windows.

---

## Project structure

```
Tech-App-Devoir-II/
├── README.md           # This file
├── requirements.txt    # Python dependencies (CPU + optional CUDA)
├── main.py             # Entry point: CLI and pipeline orchestration
├── config.py           # Central configuration (paths, hyperparameters, device)
├── data.py             # Data loading, preprocessing, and dataset utilities
├── train.py            # Training loop and model fitting
├── evaluate.py         # Evaluation, metrics, and reporting
├── .gitignore
└── .github/workflows/  # CI (e.g. SonarQube)
```

Each script is documented at the top with its role and internal structure (sections, main functions). No business logic is implemented in the stubs—only comments and minimal scaffolding.

---

## Prerequisites

- **Python**: 3.10+ recommended (3.11 or 3.12 supported).
- **CUDA** (optional): For GPU acceleration, install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and matching cuDNN. PyTorch will use it if available; otherwise it falls back to CPU.

---

## Setup (venv, Linux & Windows)

### 1. Clone and enter the project

```bash
cd /path/to/Tech-App-Devoir-II
```

### 2. Create and activate a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip (recommended)

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

**CPU only (works everywhere):**

```bash
pip install -r requirements.txt
```

**With CUDA (Linux / Windows):**

- Install PyTorch with the correct CUDA version from [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install the rest:

  **Example (CUDA 12.x):**

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```

  **Example (CUDA 11.8):**

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```

- Adjust `cu121` / `cu118` to match your installed CUDA version. The rest of `requirements.txt` is CPU-only and shared across platforms.

---

## Running the project

With the venv activated:

```bash
python main.py
```

Use the comments in `main.py` and in each script to implement CLI arguments (e.g. `--train`, `--evaluate`, `--data-path`) and to wire config, data, train, and evaluate together.

---

## Keeping it simple (KISS)

- Single venv (`.venv`), one `requirements.txt` for all non-PyTorch deps.
- PyTorch/CUDA is installed separately so each machine can choose CPU or the right CUDA build.
- No extra config layers or frameworks unless the assignment requires them.
- Scripts are flat (no heavy package layout); add a `src/` package later if the assignment grows.

---

## License

See [LICENSE](LICENSE).
