"""
Startup entrypoint for Docker.
1. Seeds the default admin user from env vars if not present.
2. Starts uvicorn.
"""
import os
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def seed_admin():
    """Create default admin user if no users exist."""
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@shop.local")
    admin_password = os.environ.get("ADMIN_DEFAULT_PASSWORD", "changeme")
    # Override DATABASE_URL if set in environment
    db_url = os.environ.get("DATABASE_URL", "sqlite:///./shop.db")
    # Update database URL in app config before import
    os.environ.setdefault("DATABASE_URL", db_url)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from shop.database import Base
    from shop.models import User, ShopConfig
    from shop.services.auth import hash_password
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        existing = db.query(User).filter(User.role == "admin").first()
        if not existing:
            admin = User(
                email=admin_email,
                hashed_password=hash_password(admin_password),
                role="admin",
                is_active=True,
            )
            db.add(admin)
            db.commit()
            logger.info(f"Seeded default admin: {admin_email}")
        else:
            logger.info(f"Admin already exists: {existing.email}")
        # Ensure singleton ShopConfig row exists
        config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
        if not config:
            db.add(ShopConfig(id=1))
            db.commit()
            logger.info("Created ShopConfig singleton")


if __name__ == "__main__":
    seed_admin()
    logger.info("Starting uvicorn on 0.0.0.0:8000")
    subprocess.run(
        ["uvicorn", "shop.app:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )
