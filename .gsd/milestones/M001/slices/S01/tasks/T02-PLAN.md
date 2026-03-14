# T02: 01-foundation 02

**Slice:** S01 — **Milestone:** M001

## Description

Create the core `shop/` Python package: database engine, ORM models, password hashing service, and FastAPI dependency functions. This is the foundation all subsequent plans import from.

Purpose: Plans 03-07 all depend on `shop.models`, `shop.dependencies`, and `shop.services.auth`. Nothing can be tested until this package exists.
Output: Importable `shop/` package with working DB layer, auth service, and dependency injection chain.

## Must-Haves

- [ ] "shop package is importable: `python -c 'import shop'` succeeds"
- [ ] "SQLAlchemy engine creates all tables (users, sessions, shop_config) without error"
- [ ] "hash_password() and verify_password() work correctly with bcrypt cost 12"
- [ ] "get_current_user() dependency reads session cookie and returns User or raises redirect"
- [ ] "require_admin() dependency silently redirects engineers to /dashboard"

## Files

- `shop/__init__.py`
- `shop/app.py`
- `shop/database.py`
- `shop/models.py`
- `shop/dependencies.py`
- `shop/services/__init__.py`
- `shop/services/auth.py`
