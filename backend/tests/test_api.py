from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_health_and_market_endpoints() -> None:
    with client() as c:
        health = c.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["aave"] is False
        assets = c.get("/assets")
        assert assets.status_code == 200
        assert len(assets.json()) >= 3
        reserves = c.get("/reserves")
        assert reserves.status_code == 200
        row = reserves.json()[0]
        assert "supply_apr" in row
        assert row["protocol"] == "educational_aave_inspired_simulation"
        strategies = c.get("/strategies")
        names = {item["name"] for item in strategies.json()}
        assert {"equal_weight", "highest_yield", "ml_optimizer"} <= names


def test_predict_allocate_simulate_experiments() -> None:
    with client() as c:
        status = c.get("/model/status")
        assert status.status_code == 200
        assert status.json()["data_label"] == "synthetic_local"
        pred = c.post("/model/predict", json={})
        assert pred.status_code == 200
        assert pred.json()["disclaimer"]
        alloc = c.post("/optimizer/allocate", json={"strategy": "ml_optimizer"})
        assert alloc.status_code == 200
        weights = alloc.json()["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-5
        assert alloc.json()["used_ml_predictions"] is True
        sim = c.post(
            "/simulation/run",
            json={"steps": 8, "scenario": "high_volatility", "strategy": "equal_weight", "seed": 7},
        )
        assert sim.status_code == 200
        payload = sim.json()
        assert payload["synthetic"] is True
        assert payload["final_value"] > 0
        experiments = c.get("/experiments")
        assert experiments.status_code == 200
        assert len(experiments.json()) >= 1
        metrics = c.get("/metrics")
        assert metrics.status_code == 200
        portfolio = c.get("/portfolio")
        assert portfolio.status_code == 200


def test_unknown_strategy_rejected() -> None:
    with client() as c:
        resp = c.post("/optimizer/allocate", json={"strategy": "magic"})
        assert resp.status_code == 400
