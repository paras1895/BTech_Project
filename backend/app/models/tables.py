from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    scenario: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer)
    steps: Mapped[int] = mapped_column(Integer)
    data_label: Mapped[str] = mapped_column(String(64))
    notes: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[str] = mapped_column(Text)


class PortfolioSnapshotRow(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    strategy: Mapped[str] = mapped_column(String(64))
    expected_return: Mapped[float] = mapped_column(Float)
    risk: Mapped[float] = mapped_column(Float)
    weights_json: Mapped[str] = mapped_column(Text)
