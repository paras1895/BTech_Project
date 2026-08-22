# Backend API

The FastAPI service is an academic research API. It talks to an **educational Aave-inspired simulation**, not the Aave protocol. The default Ridge model is trained on **synthetic local series**.

Base URL (local): `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Liveness. `aave` is always false for this backend. |
| GET | `/assets` | Simulated assets from `configs/assets.yaml`. |
| GET | `/reserves` | Utilization, APRs from the interest-rate curve, liquidity, oracle price. |
| GET | `/portfolio` | Last allocation or equal-weight default. |
| GET | `/strategies` | Shared strategy names used by optimizer and simulation. |
| GET | `/metrics` | Last simulation metrics plus live TVL / utilization. |
| GET | `/model/status` | Artifact metadata and chronological-split metrics. |
| POST | `/model/predict` | Predicts next-step **supply APR**. Body: `{ "symbols": ["USDC"] }` optional. |
| POST | `/optimizer/allocate` | ML predicts (if `ml_optimizer`) then constrained weights. Never sends chain keys. |
| POST | `/simulation/run` | Seeded synthetic scenario. Body: `steps`, `scenario`, `strategy`, `seed`, `n_users`, `capital`. |
| GET | `/experiments` | Persisted simulation runs (SQLite). |

## Scenarios

Values in `configs/simulation.yaml`. All are **synthetic labels**, not historical events.

## Run

```bash
python3 -m pip install -r backend/requirements.txt
make backend
# or
./scripts/start-backend.sh
```

Tests:

```bash
make backend-test
```
