.PHONY: backend backend-test install-backend

install-backend:
	python3 -m pip install -r backend/requirements.txt

backend:
	cd backend && python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

backend-test:
	cd backend && python3 -m pytest -q

test: backend-test
