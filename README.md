Multi-agent predator-prey

[Download ZIP](https://github.com/erik-helmers/rl/archive/refs/heads/master.zip)

## Setup

### 1. Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See https://docs.astral.sh/uv/getting-started/installation/ for other methods.

### 2. Install dependencies

```bash
uv sync
uv pip install -e .
```

### 3. Run tests

```bash
uv run pytest
```


## Project structure

```
flock/
  env/          # Environment, physics, rewards 
  train/        # PPO, policy networks, training loop
  web/          # Visualization frontend (TBD)
tests/
notebooks/
```

