import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from shop.database import Base, engine
from shop.middleware.setup_guard import SetupGuardMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

templates = Jinja2Templates(directory="shop/templates")


def create_app() -> FastAPI:
    app = FastAPI(title="Delta Preservation")
    # Create all DB tables on startup (Base.metadata.create_all)
    Base.metadata.create_all(bind=engine)
    # Middleware (outermost registered = executes last)
    app.add_middleware(SetupGuardMiddleware)
    # Static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # static/ dir may not exist in tests
    # Routers registered by subsequent plans (03-05)
    return app


app = create_app()
