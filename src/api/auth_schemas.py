# src/api/auth_schemas.py

"""Pydantic models for the auth API (Phase 14). Kept separate from schemas.py
to isolate the auth contract (and its email-validator dependency)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=200)
    name: Optional[str] = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=200)


class UserPublic(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    provider: str
    email_verified: bool = False
    created_at: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserPublic
    request_id: str


class MeResponse(BaseModel):
    user: UserPublic
    request_id: str


class ResetRequest(BaseModel):
    email: EmailStr


class ResetConfirm(BaseModel):
    token: str = Field(..., min_length=8)
    password: str = Field(..., min_length=8, max_length=200)


class MessageResponse(BaseModel):
    message: str
    # Dev-only: present when SMTP isn't configured so the flow is testable.
    reset_link: Optional[str] = None
    request_id: str


class UserStatsResponse(BaseModel):
    stats: dict
    request_id: str


class ProvidersAvailable(BaseModel):
    """Which login methods the deployment has configured (drives the UI)."""

    password: bool = True
    google: bool = False
    github: bool = False
    auth_required: bool = False
    request_id: str
