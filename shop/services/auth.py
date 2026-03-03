import secrets
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher
from shop.models import User, UserSession

# Cost factor 12 per locked decision; bcrypt must be pinned <5.0.0 in pyproject.toml
password_hash = PasswordHash([BcryptHasher(rounds=12)])


def hash_password(plain: str) -> str:
    return password_hash.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return password_hash.verify(plain, hashed)


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    session = UserSession(
        id=token,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(hours=8),
    )
    db.add(session)
    db.commit()
    return token


def delete_session(db: Session, token: str) -> None:
    db.query(UserSession).filter(UserSession.id == token).delete()
    db.commit()
