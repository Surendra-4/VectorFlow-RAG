# src/api/routes/auth.py

"""
Authentication API (Phase 14): email/password + (OAuth added in the same router).

Stateless JWT — the client stores the access token and sends it as
``Authorization: Bearer <token>``. Logout is therefore client-side (drop the
token); the endpoint exists for symmetry. Password reset is opaque: the request
endpoint behaves identically whether or not the email exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
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
from src.auth import oauth, service
from src.auth.emails import reset_link, send_password_reset
from src.auth.security import create_access_token, generate_state
from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["auth"], prefix="/auth")

_OAUTH_STATE_COOKIE = "vfr_oauth_state"


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
    # get_current_user returns a detached instance — re-attach by id.
    u = service.get_user(db, user.id)
    stats = service.ensure_stats(db, u)
    db.commit()
    return UserStatsResponse(stats=stats.to_dict(), request_id=request_id)


@router.post("/me/stats/reset", response_model=UserStatsResponse)
def reset_my_stats(
    user=Depends(get_current_user),
    db: Session = Depends(get_db_session),
    request_id: str = Depends(get_request_id),
) -> UserStatsResponse:
    u = service.get_user(db, user.id)
    stats = service.reset_user_stats(db, u)
    db.commit()
    logger.info("User %s reset their dashboard stats", u.email)
    return UserStatsResponse(stats=stats.to_dict(), request_id=request_id)


# --------------------------------------------------------------------------- #
# OAuth (Google / GitHub) — authorization-code flow (Phase 14c)
# --------------------------------------------------------------------------- #


@router.get("/{provider}")
def oauth_start(provider: str):
    """Redirect the browser to the provider's consent screen.

    A random ``state`` is set as a SameSite=Lax, HttpOnly cookie and echoed in
    the authorize URL; the callback verifies they match (CSRF protection).
    """
    if not oauth.supported(provider):
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    if not oauth.is_configured(provider):
        raise HTTPException(status_code=503, detail=f"{provider} sign-in is not configured.")

    state = generate_state()
    url = oauth.authorize_url(provider, state)
    resp = RedirectResponse(url, status_code=307)
    secure = get_settings().auth.public_base_url.startswith("https")
    resp.set_cookie(
        _OAUTH_STATE_COOKIE, state, max_age=600, httponly=True,
        secure=secure, samesite="lax", path=f"/api/v1/auth/{provider}/callback",
    )
    return resp


@router.get("/{provider}/callback")
def oauth_callback(
    provider: str,
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db_session),
):
    """Handle the provider redirect: verify state, exchange code, upsert the
    user, then bounce back to the frontend with a JWT in the URL fragment
    (fragments never reach servers, so the token stays out of access logs)."""
    cfg = get_settings().auth
    front = cfg.frontend_url.rstrip("/")

    def _fail(reason: str) -> RedirectResponse:
        logger.warning("OAuth %s failed: %s", provider, reason)
        return RedirectResponse(f"{front}/login?error={reason}", status_code=307)

    if error:
        return _fail("access_denied")
    if not oauth.supported(provider) or not oauth.is_configured(provider):
        return _fail("provider_unavailable")
    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE)
    if not state or not cookie_state or state != cookie_state:
        return _fail("state_mismatch")
    if not code:
        return _fail("missing_code")

    try:
        token = oauth.exchange_code(provider, code)
        info = oauth.fetch_user(provider, token)
    except oauth.OAuthError as exc:
        return _fail("oauth_error")

    user = service.upsert_oauth_user(
        db, email=info.email, provider=provider, name=info.name, avatar_url=info.avatar_url
    )
    service.record_event(db, user)
    db.commit()
    db.refresh(user)

    jwt_token = create_access_token(user.id)
    # Fragment carries the token to the SPA without hitting the server/logs.
    resp = RedirectResponse(f"{front}/auth/callback#access_token={jwt_token}", status_code=307)
    resp.delete_cookie(_OAUTH_STATE_COOKIE, path=f"/api/v1/auth/{provider}/callback")
    return resp
