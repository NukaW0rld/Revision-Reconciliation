from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from shop.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)  # "admin" | "engineer"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    runs: Mapped[list["Run"]] = relationship(back_populates="reviewer")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role!r}>"


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # secrets.token_urlsafe(32)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    user: Mapped["User"] = relationship(back_populates="sessions")


class ShopConfig(Base):
    __tablename__ = "shop_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shop_name: Mapped[str] = mapped_column(String, default="")
    setup_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    column_mapping: Mapped[dict] = mapped_column(JSON, default=dict)
    wizard_step: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<ShopConfig setup_complete={self.setup_complete} wizard_step={self.wizard_step}>"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    part_number: Mapped[str] = mapped_column(String)
    rev_a_label: Mapped[str] = mapped_column(String)
    rev_b_label: Mapped[str] = mapped_column(String)
    customer: Mapped[str] = mapped_column(String)
    job_number: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued/running/completed/failed/warning
    current_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    current_stage_index: Mapped[int] = mapped_column(Integer, default=0)
    failure_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    failure_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    warning_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # "revB_balloon" | "low_confidence"
    confidence_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_dir: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revA_path: Mapped[str] = mapped_column(String)
    revB_path: Mapped[str] = mapped_column(String)
    form3_path: Mapped[str] = mapped_column(String)
    revA_page: Mapped[int] = mapped_column(Integer, default=0)
    revB_page: Mapped[int] = mapped_column(Integer, default=0)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewer_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewer: Mapped[Optional["User"]] = relationship(back_populates="runs")
    alerts: Mapped[list["RunAlert"]] = relationship(back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Run id={self.id} part_number={self.part_number!r} status={self.status!r}>"


class RunAlert(Base):
    __tablename__ = "run_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    message: Mapped[str] = mapped_column(String)
    failure_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    run: Mapped["Run"] = relationship(back_populates="alerts")
    user: Mapped["User"] = relationship()

    def __repr__(self) -> str:
        return f"<RunAlert id={self.id} run_id={self.run_id} is_read={self.is_read}>"
