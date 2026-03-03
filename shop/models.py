from datetime import datetime
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
