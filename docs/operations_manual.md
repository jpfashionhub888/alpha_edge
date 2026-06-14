# AlphaEdge Operations Manual

This document outlines the standard operating procedures for managing the remote trading server.

---

## 1. Handling Long-Running Processes (`screen` Protocol)

Remote SSH connections are prone to dropping. When an SSH connection closes, any active process started in that shell is terminated. To prevent this, **always** use `screen` for any task taking more than a couple of minutes (e.g., backtests, ML model retraining, bulk data fetches).

### Common Scenarios
- **Backtesting**: Runs for multiple minutes and uses heavy memory.
- **Model Retraining**: Training random forests, LightGBM, and XGBoost models on historical data.
- **Bulk Data Fetches**: Downloading 5+ years of stock/crypto historical data.

### Basic commands

#### Start a new named screen session
```bash
screen -S backtest
```
This opens a new terminal window on the server. You can run your command inside it.

#### Detach from a running screen session
To detach and leave the process running in the background, press:
`Ctrl + A` followed by `D`

#### List all active screen sessions
```bash
screen -ls
```

#### Reattach to an existing session
```bash
screen -r backtest
```

#### Run a command in a detached session directly
If you want to start a script and immediately detach it:
```bash
screen -dmS backtest python run_backtest_v2.py --stocks
```

#### Kill/Close a session
If you are inside the session, simply type:
```bash
exit
```
If you are outside the session:
```bash
screen -XS backtest quit
```

---

## 2. Dependency Locking Procedure

During development, `requirements.txt` has its version pins stripped to resolve dependency conflicts across multiple platforms (e.g. tree-based classifiers and deep learning libraries).

Once the trading droplet is stable and functioning correctly, follow this procedure to lock the dependencies so that future deployments on fresh servers are guaranteed to be identical:

### Generate the Lock File (on the working server)
```bash
source venv/bin/activate
pip freeze > requirements-lock.txt
```

### Install from the Lock File (on a fresh server)
```bash
source venv/bin/activate
pip install -r requirements-lock.txt
```
This avoids fetching newer, potentially breaking packages.
