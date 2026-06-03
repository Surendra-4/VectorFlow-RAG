# src/api/routes/auth.py

"""
Authentication API (Phase 14): email/password + (OAuth added in the same router).

Stateless JWT — the client stores the access token and sends it as
``Authorization: Bearer <token>``. Logout is therefore client-side (drop the
token); the endpoint exists for symmetry. Password reset is opaque: the request
endpoint behaves identically whether or not the email exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.auth_schemas import (
    LoginRequest,
    MeResponse,
    MessageResponse,
    ProvidersAvailable,
    ResetConfirm,
    ResetRequest,
    SignupRequest,
    TokenResponse,
    UserPublic,
    UserStatsResponse,
)
from src.api.dependencies import get_current_user, get_db_session, get_request_id
from src.auth import service
from src.auth.emails import reset_link, send_password_reset
from src.auth.security import create_access_token
from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["auth"], prefix="/auth")


def _token_response(user, request_id: str) -> TokenResponse:
    cfg = get_settings().auth
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=cfg.jwt_expiry_minutes * 60,
        user=UserPublic(**user.to_public()),
        request_id=request_id,
    )


@router.get("/providers", response_model=ProvidersAvailable)
def providers(request_id: str = Depends(get_request_id)) -> ProvidersAvailable:
    cfg = get_settings().auth
    return ProvidersAvailable(
        password=True,
        google=bool(cfg.google_client_id and cfg.google_client_secret),
        github=bool(cfg.github_client_id and cfg.github_client_secret),
        auth_required=cfg.required,
        request_id=request_id,
    )


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(
    req: SignupRequest,
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> TokenResponse:
    user = service.create_local_user(db, email=req.email, password=req.password, name=req.name)
    db.commit()
    db.refresh(user)
    return _token_response(user, request_id)


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> TokenResponse:
    user = service.authenticate(db, email=req.email, password=req.password)
    service.record_event(db, user)  # touch last_active
    db.commit()
    return _token_response(user, request_id)


@router.get("/me", response_model=MeResponse)
def me(
    user=Depends(get_current_user),
    request_id: str = Depends(get_request_id),
) -> MeResponse:
    return MeResponse(user=UserPublic(**user.to_public()), request_id=request_id)


@router.post("/logout", response_model=MessageResponse)
def logout(request_id: str = Depends(get_request_id)) -> MessageResponse:
    # JWT is stateless — the client discards the token. Endpoint kept for symmetry.
    return MessageResponse(message="Signed out.", request_id=request_id)


@router.post("/reset/request", response_model=MessageResponse)
def reset_request(
    req: ResetRequest,
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> MessageResponse:
    token = service.begin_password_reset(db, req.email)
    db.commit()
    dev_link = None
    if token is not None:
        emailed = send_password_reset(req.email, token)
        if not emailed:
            dev_link = reset_link(token)  # SMTP not configured → surface for dev
    # Opaque response regardless of whether the account exists.
    return MessageResponse(
        message="If an account exists for that email, a reset link has been sent.",
        reset_link=dev_link,
        request_id=request_id,
    )


@router.post("/reset/confirm", response_model=TokenResponse)
def reset_confirm(
    req: ResetConfirm,
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> TokenResponse:
    user = service.complete_password_reset(db, token=req.token, new_password=req.password)
    db.commit()
    db.refresh(user)
    return _token_response(user, request_id)


@router.get("/me/stats", response_model=UserStatsResponse)
def my_stats(
    user=Depends(get_current_user),
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> UserStatsResponse:
    stats = service.ensure_stats(db, user)
    db.commit()
    return UserStatsResponse(stats=stats.to_dict(), request_id=request_id)


@router.post("/me/stats/reset", response_model=UserStatsResponse)
def reset_my_stats(
    user=Depends(get_current_user),
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> UserStatsResponse:
    stats = service.reset_user_stats(db, user)
    db.commit()
    logger.info("User %s reset their dashboard stats", user.email)
    return UserStatsResponse(stats=stats.to_dict(), request_id=request_id)
