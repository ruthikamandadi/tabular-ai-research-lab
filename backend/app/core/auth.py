"""Cookie- and bearer token-based account authentication and authorization dependencies."""
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core.config import FRONTEND_ORIGINS
from app.core.database import AuthSessionRecord, UserRecord, get_db

logger = logging.getLogger("app.auth")

router = APIRouter(prefix="/api/auth", tags=["authentication"])
SESSION_COOKIE = "tabular_session"
SESSION_DURATION = timedelta(days=30)
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_ATTEMPTS = 5
_LOGIN_FAILURES = {}
_LOGIN_LOCK = threading.Lock()
_PASSWORD_PATTERN = re.compile(r"^scrypt\$([0-9a-f]{32})\$([0-9a-f]{128})$")


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)
    confirm_password: str = Field(min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        normalized = value.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
            raise ValueError("Enter a valid email address.")
        return normalized

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value):
        name = value.strip()
        if not name:
            raise ValueError("Name is required.")
        return name

    def validate_password_confirmation(self):
        if self.password != self.confirm_password:
            raise HTTPException(422, "Passwords do not match.")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        return value.strip().lower()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return f"scrypt${salt.hex()}${derived.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    match = _PASSWORD_PATTERN.fullmatch(password_hash)
    if match is None:
        return False
    salt = bytes.fromhex(match.group(1))
    expected = bytes.fromhex(match.group(2))
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return hmac.compare_digest(actual, expected)


def _get_allowed_origins() -> set[str]:
    cors_env = os.getenv("CORS_ORIGINS")
    raw = cors_env.split(",") if cors_env else FRONTEND_ORIGINS
    origins = set()
    for item in raw:
        cleaned = item.strip().rstrip("/")
        if cleaned:
            origins.add(cleaned)
    return origins


def _request_origin_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    normalized_origin = origin.rstrip("/")
    allowed = _get_allowed_origins()
    base = str(request.base_url).rstrip("/")
    return normalized_origin in allowed or normalized_origin == base


def _client_keys(request: Request, email: str):
    remote = request.client.host if request.client else "unknown"
    email_key = hashlib.sha256(email.encode("utf-8")).hexdigest()
    return (f"ip:{remote}", f"email:{email_key}")


def _check_login_rate_limit(request: Request, email: str):
    now = time.monotonic()
    keys = _client_keys(request, email)
    with _LOGIN_LOCK:
        for key in keys:
            started, count = _LOGIN_FAILURES.get(key, (now, 0))
            if now - started >= LOGIN_WINDOW_SECONDS:
                _LOGIN_FAILURES[key] = (now, 0)
                continue
            if count >= LOGIN_MAX_ATTEMPTS:
                raise HTTPException(429, "Too many login attempts. Try again in 15 minutes.")


def _record_login_result(request: Request, email: str, success: bool):
    now = time.monotonic()
    keys = _client_keys(request, email)
    with _LOGIN_LOCK:
        if success:
            for key in keys:
                _LOGIN_FAILURES.pop(key, None)
            return
        for key in keys:
            started, count = _LOGIN_FAILURES.get(key, (now, 0))
            if now - started >= LOGIN_WINDOW_SECONDS:
                started, count = now, 0
            _LOGIN_FAILURES[key] = (started, count + 1)


def _user_payload(user: UserRecord):
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _create_session(response: Response, user: UserRecord, db: Session) -> str:
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(UTC) + SESSION_DURATION
    db.add(AuthSessionRecord(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token_hash=hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
        expires_at=expires_at,
    ))
    db.commit()
    from app.core.config import COOKIE_SECURE

    response.set_cookie(
        SESSION_COOKIE,
        raw_token,
        max_age=int(SESSION_DURATION.total_seconds()),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return raw_token


@router.post("/register", status_code=201)
def register(body: RegisterRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    body.validate_password_confirmation()
    if not _request_origin_allowed(request):
        logger.warning("Authentication rejected: origin not allowed")
        raise HTTPException(403, "Request origin is not allowed.")
    if db.query(UserRecord).filter(UserRecord.email == body.email).first():
        raise HTTPException(409, "An account with this email already exists.")

    user = UserRecord(
        id=str(uuid.uuid4()),
        name=body.name,
        email=body.email,
        password_hash=_hash_password(body.password),
    )
    db.add(user)
    try:
        db.commit()
    except Exception:
        db.rollback()
        if db.query(UserRecord).filter(UserRecord.email == body.email).first():
            raise HTTPException(409, "An account with this email already exists.")
        raise
    db.refresh(user)
    token = _create_session(response, user, db)
    return {"user": _user_payload(user), "token": token}


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    if not _request_origin_allowed(request):
        logger.warning("Authentication rejected: origin not allowed")
        raise HTTPException(403, "Request origin is not allowed.")
    _check_login_rate_limit(request, body.email)
    user = db.query(UserRecord).filter(UserRecord.email == body.email).first()
    valid = user is not None and _verify_password(body.password, user.password_hash)
    _record_login_result(request, body.email, valid)
    if not valid:
        raise HTTPException(401, "Email or password is incorrect.")
    token = _create_session(response, user, db)
    return {"user": _user_payload(user), "token": token}


def _extract_token(request: Request) -> str | None:
    # 1. Authorization: Bearer <token>
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            if token:
                return token
    # 2. Query parameter ?token= (for direct browser file downloads / HTML report links)
    query_token = request.query_params.get("token")
    if query_token:
        token = query_token.strip()
        if token:
            return token
    # 3. Session cookie (for backwards compatibility with local development)
    cookie_token = request.cookies.get(SESSION_COOKIE)
    if cookie_token:
        return cookie_token.strip()
    return None


def require_authenticated_user(request: Request, db: Session = Depends(get_db)):
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not _request_origin_allowed(request):
        logger.warning("Authentication rejected: origin not allowed")
        raise HTTPException(403, "Request origin is not allowed.")

    raw_token = _extract_token(request)
    if not raw_token:
        logger.warning("Authentication rejected: no token")
        raise HTTPException(401, "Authentication required.")

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    session = db.query(AuthSessionRecord).filter(
        AuthSessionRecord.token_hash == token_hash,
    ).first()

    if not session:
        logger.warning("Authentication rejected: token unknown")
        raise HTTPException(401, "Session expired. Please sign in again.")

    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        logger.warning("Authentication rejected: token expired")
        raise HTTPException(401, "Session expired. Please sign in again.")

    user = db.query(UserRecord).filter(UserRecord.id == session.user_id).first()
    if not user:
        logger.warning("Authentication rejected: user not found")
        raise HTTPException(401, "Authentication required.")

    request.state.user = user
    request.state.raw_token = raw_token
    return user


def request_user_id(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Authentication required.")
    return user.id


@router.get("/me")
def get_current_user(user: UserRecord = Depends(require_authenticated_user)):
    return {"user": _user_payload(user)}


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    user: UserRecord = Depends(require_authenticated_user),
):
    raw_token = getattr(request.state, "raw_token", None) or _extract_token(request)
    if raw_token:
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        db.query(AuthSessionRecord).filter(
            AuthSessionRecord.token_hash == token_hash,
            AuthSessionRecord.user_id == user.id,
        ).delete(synchronize_session=False)
        db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")
    response.delete_cookie(SESSION_COOKIE, path="/api", httponly=True, samesite="lax")
    return {"message": "Signed out."}
