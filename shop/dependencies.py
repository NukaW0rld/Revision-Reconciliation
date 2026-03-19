import logging
from datetime import timedelta
from typing import Generator
from fastapi import Depends, Request, HTTPException
from sqlalchemy.orm import Session
from shop.utils import utcnow
from shop.database import SessionLocal
from shop.models import User, UserSession

logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("session_token")
    if not token:
        raise _redirect_to_login()
    session = db.query(UserSession).filter(
        UserSession.id == token,
        UserSession.expires_at > utcnow()
    ).first()
    if not session:
        raise _redirect_to_login()
    # Sliding window: reset expiry on every authenticated request
    session.expires_at = utcnow() + timedelta(hours=8)
    db.commit()
    return session.user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        # Silent redirect — per locked decision: no 403 page, no error message
        raise HTTPException(status_code=302, headers={"Location": "/dashboard"})
    return user


def _redirect_to_login() -> HTTPException:
    return HTTPException(status_code=302, headers={"Location": "/login"})
