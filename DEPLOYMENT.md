# Deploying VectorFlow-RAG

This guide takes you from a fresh clone to a live, multi-user deployment:

- **Frontend** → [Vercel](https://vercel.com) (Next.js)
- **Backend** → [Render](https://render.com) (FastAPI)
- **Database** → Render managed **PostgreSQL** (users + per-user statistics only — never your ingested documents)

Both services connect to GitHub and redeploy on every push. The whole thing is
**environment-driven**: the exact same code that runs locally on SQLite runs in
production on Postgres — you only change configuration, never code.

---

## Contents

1. [Architecture](#1-architecture)
2. [Before you start](#2-before-you-start)
3. [Push the code to GitHub](#3-push-the-code-to-github)
4. [Deploy the backend on Render](#4-deploy-the-backend-on-render)
5. [Deploy the frontend on Vercel](#5-deploy-the-frontend-on-vercel)
6. [Connect the two (CORS + URLs)](#6-connect-the-two-cors--urls)
7. [Enable Google & GitHub sign-in](#7-enable-google--github-sign-in)
8. [Choose an answer model](#8-choose-an-answer-model)
9. [Smoke test](#9-smoke-test)
10. [Sizing & cost](#10-sizing--cost)
11. [Operational notes](#11-operational-notes)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture

```
            ┌────────────────────┐         ┌───────────────────────────┐
  Browser ──┤  Vercel (Next.js)  ├──HTTPS──┤  Render (FastAPI)         │
            │  the web UI        │  fetch  │  hybrid RAG + auth API     │
            └────────────────────┘         └───────────┬───────────────┘
                                                        │ SQLAlchemy
                                                        ▼
                                            ┌───────────────────────────┐
                                            │  Render PostgreSQL         │
                                            │  users + per-user stats    │
                                            └───────────────────────────┘
```

- The browser holds a **JWT** (issued by the backend) in `localStorage` and
  sends it as `Authorization: Bearer …` on every API call.
- **Google / GitHub** sign-in uses the OAuth 2.0 authorization-code flow,
  handled entirely by the backend; the browser only follows redirects.
- **Postgres** stores accounts and each user's counters (searches, questions,
  documents, …). **Ingested documents are never stored in the database** — they
  live in the vector index on the backend instance (see
  [Operational notes](#11-operational-notes) for what persists across restarts).

---

## 2. Before you start

Create free accounts if you don't have them:

- [ ] [GitHub](https://github.com) — the repo is already here.
- [ ] [Render](https://render.com) — backend + database.
- [ ] [Vercel](https://vercel.com) — frontend.
- [ ] *(optional)* [Google Cloud Console](https://console.cloud.google.com) and/or a GitHub OAuth App for social sign-in.
- [ ] *(optional)* An API key for a hosted LLM (OpenAI / Anthropic / Google Gemini / Groq / OpenRouter) — the answer model. See [step 8](#8-choose-an-answer-model).

**Pick your two URLs up front.** Most steps reference them, and OAuth needs them
to match exactly. They're predictable from the service names you choose:

| What | Example | You'll set it as |
|------|---------|------------------|
| Backend (Render) | `https://vectorflow-api.onrender.com` | `VFR_AUTH__PUBLIC_BASE_URL` |
| Frontend (Vercel) | `https://vectorflow.vercel.app` | `VFR_AUTH__FRONTEND_URL` |

> Render derives the backend URL from the **service name** in `render.yaml`
> (`vectorflow-api` → `https://vectorflow-api.onrender.com`). If that name is
> already taken globally, Render appends a suffix — check the real URL after
> creation and update the env vars.

The reference templates are committed for you:

- `render.yaml` — the Render Blueprint (web service + Postgres).
- `.env.production.example` — every backend variable, annotated.
- `frontend/vercel.json` + `frontend/.env.production.example` — the frontend.

---

## 3. Push the code to GitHub

Render and Vercel deploy from a branch (`main`). Make sure your latest work and
the `v1.0.0` baseline tag are on the remote:

```bash
# from the repo root, on main, with the work merged in
git push origin main
git push origin v1.0.0        # the "version 1" baseline tag
git push origin --tags        # any remaining phase tags
```

> If you work on a feature branch, fast-forward `main` to it first, then push.
> Render's Blueprint and Vercel's project both track `main` by default.

---

## 4. Deploy the backend on Render

### 4a. Create the Blueprint

1. Render dashboard → **New ▸ Blueprint**.
2. Connect your GitHub account and pick this repository.
3. Render finds `render.yaml` and shows a plan: one **web service**
   (`vectorflow-api`) + one **PostgreSQL** database (`vectorflow-db`).
4. You'll be prompted for the values marked `sync: false`. Enter your best
   guess now — you can edit them after the URLs are final:
   - `VFR_AUTH__PUBLIC_BASE_URL` = `https://vectorflow-api.onrender.com`
   - `VFR_AUTH__FRONTEND_URL` = `https://vectorflow.vercel.app`
   - `VFR_API__CORS_ORIGINS` = `["https://vectorflow.vercel.app"]`
5. Click **Apply**. Render provisions the database, then builds the web service.

The first build installs PyTorch + sentence-transformers + FAISS + Chroma and
takes **5–15 minutes**. Watch the logs; a healthy boot ends with:

```
Initializing RAGPipeline for HTTP service…
Pipeline ready (backend=chromadb, cache=none)
DB schema ensured (users, user_stats)
Uvicorn running on http://0.0.0.0:10000
```

### 4b. What `render.yaml` already handles for you

- **Start command** — `uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port $PORT`
  (the app is factory-only; this is the correct invocation).
- **`DATABASE_URL`** — wired automatically from the managed Postgres. The app
  accepts `postgres://` / `postgresql://` and normalizes the driver to
  `postgresql+psycopg2`. **You never paste a connection string.**
- **`VFR_AUTH__JWT_SECRET`** — generated by Render (strong + random).
- **`VFR_AUTH__REQUIRED=true`** — locks the data endpoints behind login.
- **Health check** at `/health`.
- **A 1 GB persistent disk** mounted at `var/` so provider API keys and the
  selected model survive restarts (see [Operational notes](#11-operational-notes)).

### 4c. Confirm the real URL

Once live, open `https://<your-service>.onrender.com/health` — you should get
`{"status":"ok", …}`. If the service name differs from `vectorflow-api`, update
`VFR_AUTH__PUBLIC_BASE_URL` (and the OAuth redirect URIs in step 7) to match.

---

## 5. Deploy the frontend on Vercel

1. Vercel dashboard → **Add New ▸ Project** → import this repository.
2. **Set the Root Directory to `frontend`.** This is the one setting people
   miss — the Next.js app lives in the `frontend/` subfolder, not the repo root.
3. Framework preset auto-detects as **Next.js**. Leave build/output at defaults.
4. Add an **Environment Variable** (scope: Production, and Preview if you use it):

   | Key | Value |
   |-----|-------|
   | `NEXT_PUBLIC_API_BASE_URL` | `https://vectorflow-api.onrender.com` |

5. **Deploy.** Vercel builds and gives you `https://<project>.vercel.app`.

> `NEXT_PUBLIC_*` variables are baked in at **build time**. If you change the
> backend URL later, you must **redeploy** the frontend (Deployments ▸ ⋯ ▸
> Redeploy) for it to take effect.

---

## 6. Connect the two (CORS + URLs)

Now that both URLs are final, make sure they reference each other. On the
**Render** service (Environment tab), confirm:

| Key | Value |
|-----|-------|
| `VFR_AUTH__PUBLIC_BASE_URL` | your real Render URL |
| `VFR_AUTH__FRONTEND_URL` | your real Vercel URL |
| `VFR_API__CORS_ORIGINS` | `["https://<your-vercel-app>.vercel.app"]` |

`VFR_API__CORS_ORIGINS` must be a **JSON array** and match the browser origin
**exactly** — scheme + host, no trailing slash. Add extra entries for custom
domains or Vercel preview URLs:

```json
["https://vectorflow.vercel.app", "https://www.yourdomain.com"]
```

Saving env vars on Render triggers a redeploy. Wait for it to go green.

---

## 7. Enable Google & GitHub sign-in

Optional — **email/password works without any of this.** Add a provider only
when you want its button to light up. The frontend shows a provider's button
only when the backend reports it as configured (`GET /api/v1/auth/providers`).

The callback URL pattern is always:

```
{VFR_AUTH__PUBLIC_BASE_URL}/api/v1/auth/{provider}/callback
```

### 7a. Google

1. [Google Cloud Console](https://console.cloud.google.com) → create/select a project.
2. **APIs & Services ▸ OAuth consent screen** → configure (External; add your
   email as a test user while in “Testing”).
3. **APIs & Services ▸ Credentials ▸ Create Credentials ▸ OAuth client ID**
   → Application type **Web application**.
4. Under **Authorized redirect URIs**, add exactly:
   ```
   https://vectorflow-api.onrender.com/api/v1/auth/google/callback
   ```
5. Copy the **Client ID** and **Client secret**, then add to Render:
   - `VFR_AUTH__GOOGLE_CLIENT_ID`
   - `VFR_AUTH__GOOGLE_CLIENT_SECRET`

### 7b. GitHub

1. [GitHub ▸ Settings ▸ Developer settings ▸ OAuth Apps](https://github.com/settings/developers) → **New OAuth App**.
2. **Homepage URL** = your Vercel URL.
3. **Authorization callback URL** = exactly:
   ```
   https://vectorflow-api.onrender.com/api/v1/auth/github/callback
   ```
4. Generate a client secret, then add to Render:
   - `VFR_AUTH__GITHUB_CLIENT_ID`
   - `VFR_AUTH__GITHUB_CLIENT_SECRET`

Save → Render redeploys → the buttons appear on `/login`.

---

## 8. Choose an answer model

VectorFlow defaults to a **local Ollama** model, which doesn't exist on Render.
Retrieval (search) works out of the box, but **answering questions needs a chat
model.** The easiest path is a hosted provider:

1. Sign in to your deployed app and open **Settings**.
2. Under **Models / Providers**, pick a provider (OpenAI, Anthropic, Gemini,
   Groq, or OpenRouter) and paste its **API key**. The key is stored
   **server-side, encrypted at rest, and never sent back to the browser**.
3. Select a chat model and save. Ask a question to confirm.

> **Make keys survive redeploys.** Provider keys live in the encrypted store on
> the mounted disk. Set `VFR_SECRET_KEY` (a Fernet key) on Render so the
> encryption key is stable across instances:
> ```bash
> python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
> ```
> Without it the disk still persists the auto-generated key, but a fresh disk
> (or no disk) means re-entering provider keys after a redeploy.

---

## 9. Smoke test

Open your Vercel URL and walk through:

- [ ] `/login` renders the split-screen with any social buttons you enabled.
- [ ] **Create account** (email/password) → you land on the app, signed in.
- [ ] *(if enabled)* **Continue with Google / GitHub** completes and returns you signed in.
- [ ] **Ingest** a short `.txt` or paste text → it reports chunks added.
- [ ] **Search** returns results with source citations.
- [ ] **Ask** streams an answer (after [step 8](#8-choose-an-answer-model)).
- [ ] **Dashboard** shows *Your activity* counting up; **Reset my statistics** zeroes your counters only.
- [ ] **Sign out** returns you to `/login`; protected routes redirect there when signed out.
- [ ] **Forgot password** sends a reset link (emailed if SMTP is set, otherwise printed in the Render logs).

---

## 10. Sizing & cost

The backend loads a local embedding model (~1 GB resident), so RAM is the
constraint:

| Component | Plan | Notes |
|-----------|------|-------|
| Render web service | **Standard (2 GB)** recommended | `free`/`starter` (512 MB) are **OOM-killed** during model load. The symptom is the worker dying right after `Initializing RAGPipeline`. |
| Render Postgres | `free` to start | Render expires free databases — move to a paid plan before you depend on it. |
| Vercel | **Hobby (free)** | Fine for this frontend. |

To trim cost: a smaller embedding model lowers the RAM floor (set
`VFR_EMBEDDER__MODEL_NAME`), and Render's free web service works for a quick
demo if it fits in memory — but it cold-starts after inactivity (first request
takes ~30–60 s) and has no persistent disk.

---

## 11. Operational notes

**What persists, and what doesn't.** Render's filesystem is ephemeral except
for mounted disks.

| Data | Where | Survives redeploy? |
|------|-------|--------------------|
| User accounts + per-user stats | Postgres | ✅ yes |
| Provider API keys + selected model | encrypted store on the `var/` disk | ✅ yes (with the mounted disk) |
| Ingested documents / vector index | `indices/` on the instance | ❌ no — re-ingest after a restart |

This matches the design goal: **only credentials and statistics are durable;
documents are not stored in the database.** If you *want* ingested documents to
survive restarts too, point the index at the mounted disk:

```
VFR_VECTOR_STORE__PERSIST_DIRECTORY=/opt/render/project/src/var/chroma_db
```

**Cold starts.** On Render's free/idle tiers the service sleeps; the first
request wakes it and pays the model-load cost. Paid instances stay warm.

**Reset statistics.** Each user resets only their own counters from the
dashboard (*Reset my statistics*). There's nothing to configure — it calls
`POST /api/v1/auth/me/stats/reset`.

**Security defaults that production already gets, for free:**
- The OAuth state cookie is marked `Secure` automatically because
  `VFR_AUTH__PUBLIC_BASE_URL` starts with `https`.
- Passwords are bcrypt-hashed; the hash is never returned by any endpoint.
- JWTs are signed with your `VFR_AUTH__JWT_SECRET`; rotating it invalidates all
  existing sessions.

**OCR ingestion** needs the system `tesseract` binary, which Render's native
Python runtime doesn't include. Text/PDF/DOCX/CSV/XLSX/JSON ingestion all work
without it. For image OCR in production, deploy the backend as a Docker service
that `apt-get install`s `tesseract-ocr`.

---

## 12. Troubleshooting

**CORS error in the browser console** (`blocked by CORS policy`)
→ `VFR_API__CORS_ORIGINS` doesn't match the frontend origin. It must be a JSON
array containing the exact `https://…` origin, no trailing slash. Redeploy the
backend after changing it.

**`redirect_uri_mismatch` from Google / GitHub**
→ The redirect URI registered with the provider must equal
`{VFR_AUTH__PUBLIC_BASE_URL}/api/v1/auth/{provider}/callback` **character for
character**. Re-check the scheme, host, and path; update whichever side is wrong.

**Login works but every API call returns 401**
→ The frontend can't reach the backend or the token isn't attached. Confirm
`NEXT_PUBLIC_API_BASE_URL` points at the backend and the frontend was redeployed
after setting it. A 401 also clears the stored token by design — you'll be
bounced to `/login`.

**Backend deploy goes live then crashes / restarts**
→ Almost always out-of-memory on a 512 MB plan. Upgrade the web service to
Standard (2 GB). See [Sizing & cost](#10-sizing--cost).

**“DB init skipped/failed” in the logs**
→ The service started but couldn't reach Postgres. Confirm the database is
attached and `DATABASE_URL` is present in the web service's environment (the
Blueprint wires this; a manually-created service needs it added).

**Questions never answer / time out**
→ No chat model is configured. Complete [step 8](#8-choose-an-answer-model).

**Social buttons don't appear on `/login`**
→ That provider isn't configured. Verify both the client-ID and client-secret
env vars are set and the backend redeployed; check
`GET /api/v1/auth/providers` returns `true` for it.
