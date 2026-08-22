# Backend

FastAPI service for the research platform. It drives an **educational Aave-inspired simulation**, a risk engine, a constrained allocator, baseline strategies, and a Ridge model trained on **synthetic local data**.

This is not the Aave protocol and does not use real funds.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Docs: `/docs` (Swagger) and `/health`.
