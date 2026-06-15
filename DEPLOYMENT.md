# Deploying VectorFlow-RAG

This app loads PyTorch, sentence-transformers, FAISS and a local embedding
model — together ~1 GB of RAM, which blows past the free tier of most cloud
hosts. So instead of renting a big server, we **run the backend on your own
machine** and expose it to the internet through a **Cloudflare Tunnel**. The
lightweight frontend stays on **Vercel**, and accounts/stats live in a small
**managed Postgres** database.

- **Frontend** → [Vercel](https://vercel.com) (Next.js)
- **Backend** → your PC (FastAPI), published via **Cloudflare Tunnel** at a stable hostname like `https://api.yourdomain.com`
- **Database** → managed **PostgreSQL** ([Neon](https://neon.tech) free tier) — users + per-user statistics only, never your ingested documents

The whole thing is **environment-driven**: the same code you run locally is what
serves traffic — you only change configuration.

> The trade-off: your backend is online **only while your PC is on and the
> tunnel is running**. The frontend (Vercel) and the database (Neon) are always
> up; the API answers when your machine does. Run the tunnel as a background
> service ([step 6](#6-keep-it-running)) so it survives reboots.

---

## Contents

1. [Architecture](#1-architecture)
2. [Before you start](#2-before-you-start)
3. [Provision the database](#3-provision-the-database-neon)
4. [Configure & run the backend](#4-configure--run-the-backend-on-your-pc)
5. [Create the Cloudflare Tunnel](#5-create-the-cloudflare-tunnel)
6. [Keep it running](#6-keep-it-running)
7. [Deploy the frontend on Vercel](#7-deploy-the-frontend-on-vercel)
8. [Enable Google & GitHub sign-in](#8-enable-google--github-sign-in)
9. [Choose an answer model](#9-choose-an-answer-model)
10. [Smoke test](#10-smoke-test)
11. [Operational notes](#11-operational-notes)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture

```
        ┌────────────────────┐        ┌──────────────────────┐
Browser ┤  Vercel (Next.js)  ├─HTTPS─►│  Cloudflare edge      │
        │  the web UI        │  fetch │  api.yourdomain.com   │
        └────────────────────┘        └──────────┬───────────┘
                                                  │ encrypted tunnel
                                                  ▼
                                    ┌──────────────────────────┐
                                    │  YOUR PC                  │
                                    │  cloudflared ─► uvicorn   │
                                    │  FastAPI + RAG (PyTorch,  │
                                    │  FAISS, embeddings)       │
                                    └──────────┬───────────────┘
                                               │ SQLAlchemy (TLS)
                                               ▼
                                    ┌──────────────────────────┐
                                    │  Neon PostgreSQL (cloud)  │
                                    │  users + per-user stats   │
                                    └──────────────────────────┘
```

- `cloudflared` opens an **outbound** connection to Cloudflare, so there are
  **no router ports to open** and your home IP is never exposed.
- The browser holds a **JWT** in `localStorage` and sends it as
  `Authorization: Bearer …`. With `VFR_AUTH__REQUIRED=true`, the data endpoints
  reject anonymous calls.
- **Postgres** stores only accounts and counters. Ingested documents stay in
  the vector index on your PC and are never written to the database.

---

## 2. Before you start

Accounts / tools you'll need (all free except the domain):

- [ ] [GitHub](https://github.com) — the repo (Vercel deploys from it).
- [ ] [Vercel](https://vercel.com) — frontend hosting.
- [ ] [Cloudflare](https://dash.cloudflare.com/sign-up) account **+ a domain you control.** The account and DNS are free; you must own a domain and point it at Cloudflare's nameservers (Cloudflare walks you through this when you "Add a site"). A cheap `.com`/`.dev` is fine.
- [ ] [Neon](https://neon.tech) — managed Postgres (free tier is plenty).
- [ ] [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) on your PC. macOS: `brew install cloudflared`.
- [ ] Python deps installed: from the repo root, `pip install -r requirements.txt`.
- [ ] *(optional)* Google / GitHub OAuth apps for social sign-in — [step 8](#8-enable-google--github-sign-in).
- [ ] *(optional)* An API key for a hosted LLM (OpenAI / Anthropic / Gemini / Groq / OpenRouter) — the answer model, [step 9](#9-choose-an-answer-model).

**Pick your two URLs up front** — most steps reference them and OAuth needs them
to match exactly:

| What | Example | Set as |
|------|---------|--------|
| Backend (your tunnel) | `https://api.yourdomain.com` | `VFR_AUTH__PUBLIC_BASE_URL` |
| Frontend (Vercel) | `https://vectorflow.vercel.app` | `VFR_AUTH__FRONTEND_URL` |

Committed reference templates:

- `.env.production.example` — every backend variable, annotated (you copy it to `.env`).
- `deploy/cloudflared/config.example.yml` — the tunnel config.
- `frontend/.env.production.example` + `frontend/vercel.json` — the frontend.

---

## 3. Provision the database (Neon)

1. Create a [Neon](https://neon.tech) project (pick a region near you).
2. Create a database named `vectorflow` (or use the default).
3. Copy the **connection string**. It looks like:
   ```
   postgresql://user:password@ep-xxx-123.us-east-2.aws.neon.tech/vectorflow?sslmode=require
   ```
   Keep the `?sslmode=require` — Neon requires TLS. The app normalizes the
   driver to `psycopg2` automatically and creates its tables on first boot.

> Prefer [Supabase](https://supabase.com) or another Postgres host? Any works —
> just paste its connection string as `DATABASE_URL` in the next step.

---

## 4. Configure & run the backend on your PC

1. From the repo root, create your env file:
   ```bash
   cp .env.production.example .env
   ```
2. Edit `.env` and set, at minimum:
   - `DATABASE_URL` — the Neon string from step 3.
   - `VFR_AUTH__JWT_SECRET` — `python -c "import secrets; print(secrets.token_urlsafe(48))"`
   - `VFR_AUTH__PUBLIC_BASE_URL` — `https://api.yourdomain.com` (your tunnel hostname).
   - `VFR_AUTH__FRONTEND_URL` — your Vercel URL.
   - `VFR_API__CORS_ORIGINS` — `["https://<your-vercel-app>.vercel.app"]`.
   - `VFR_SECRET_KEY` — `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` (encrypts stored provider keys).

   `.env` is gitignored, so these secrets never leave your machine.
3. Start the server (it binds to `127.0.0.1:8000` and auto-loads `.env`):
   ```bash
   python -m src.api
   ```
   First boot downloads the embedding model and ends with:
   ```
   Pipeline ready (backend=chromadb, cache=none)
   DB schema ensured (users, user_stats)
   Uvicorn running on http://127.0.0.1:8000
   ```
4. Sanity-check locally: `curl http://127.0.0.1:8000/health` → `{"status":"ok",…}`.

Leave it running. Next we put it on the internet.

---

## 5. Create the Cloudflare Tunnel

Run these once, from any directory. They create a **named** tunnel with a stable
hostname — so the URL never changes, even across restarts.

```bash
brew install cloudflared                      # macOS (see CF docs for other OSes)

cloudflared tunnel login                      # opens a browser; pick your domain
cloudflared tunnel create vectorflow          # prints a tunnel UUID + writes credentials
cloudflared tunnel route dns vectorflow api.yourdomain.com   # creates the DNS record
```

Now create the config file. Copy the template and edit the three `REPLACE_`
values:

```bash
cp deploy/cloudflared/config.example.yml ~/.cloudflared/config.yml
```

`~/.cloudflared/config.yml` should end up like:

```yaml
tunnel: vectorflow
credentials-file: /Users/you/.cloudflared/<TUNNEL-UUID>.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Start the tunnel (in a second terminal, with the backend still running):

```bash
cloudflared tunnel run vectorflow
```

Verify from anywhere: `https://api.yourdomain.com/health` returns
`{"status":"ok",…}`. The backend is now live on the internet.

---

## 6. Keep it running

`cloudflared tunnel run` stops when you close the terminal. To keep the tunnel
up across reboots, install it as a background service (it reads the same
`~/.cloudflared/config.yml`):

```bash
sudo cloudflared service install        # macOS: registers a launchd service
```

Manage it with `sudo launchctl kickstart -k system/com.cloudflare.cloudflared`
(restart) or your OS's service tools.

For the **backend** itself, keep `python -m src.api` running however you prefer —
a dedicated terminal/`tmux`, a `launchd`/`systemd` unit, or `pm2 start "python -m src.api"`.
The API is reachable only while this process is up.

---

## 7. Deploy the frontend on Vercel

1. Push the repo to GitHub if you haven't:
   ```bash
   git push origin main
   ```
2. Vercel dashboard → **Add New ▸ Project** → import this repository.
3. **Set the Root Directory to `frontend`.** The Next.js app lives in the
   `frontend/` subfolder — this is the setting people miss.
4. Framework auto-detects as **Next.js**. Leave build/output at defaults.
5. Add an **Environment Variable** (scope: Production):

   | Key | Value |
   |-----|-------|
   | `NEXT_PUBLIC_API_BASE_URL` | `https://api.yourdomain.com` |

6. **Deploy.** You get `https://<project>.vercel.app`.

> `NEXT_PUBLIC_*` is baked in at **build time** — change the backend URL later and
> you must **redeploy** the frontend (Deployments ▸ ⋯ ▸ Redeploy).

If your final Vercel URL differs from what you guessed, update
`VFR_AUTH__FRONTEND_URL` and `VFR_API__CORS_ORIGINS` in the backend `.env` and
restart `python -m src.api`.

---

## 8. Enable Google & GitHub sign-in

Optional — **email/password works without any of this.** The frontend shows a
provider's button only when the backend reports it configured
(`GET /api/v1/auth/providers`). The callback URL is always:

```
{VFR_AUTH__PUBLIC_BASE_URL}/api/v1/auth/{provider}/callback
```

### Google
1. [Google Cloud Console](https://console.cloud.google.com) → create/select a project.
2. **OAuth consent screen** → configure (External; add yourself as a test user while in “Testing”).
3. **Credentials ▸ Create Credentials ▸ OAuth client ID** → **Web application**.
4. **Authorized redirect URIs** → add exactly:
   ```
   https://api.yourdomain.com/api/v1/auth/google/callback
   ```
5. Put the Client ID/secret in `.env` as `VFR_AUTH__GOOGLE_CLIENT_ID` / `VFR_AUTH__GOOGLE_CLIENT_SECRET`, restart the backend.

### GitHub
1. [GitHub ▸ Settings ▸ Developer settings ▸ OAuth Apps](https://github.com/settings/developers) → **New OAuth App**.
2. **Homepage URL** = your Vercel URL.
3. **Authorization callback URL** = exactly:
   ```
   https://api.yourdomain.com/api/v1/auth/github/callback
   ```
4. Put the Client ID/secret in `.env` as `VFR_AUTH__GITHUB_CLIENT_ID` / `VFR_AUTH__GITHUB_CLIENT_SECRET`, restart the backend.

---

## 9. Choose an answer model

VectorFlow defaults to a local **Ollama** model. Retrieval (search) works out of
the box; **answering questions needs a chat model.** Two options:

- **Local Ollama** (free, private, no API key): install [Ollama](https://ollama.com),
  `ollama pull llama3.2`, and it's already running at `http://localhost:11434` next
  to your backend — set the model in **Settings**.
- **Hosted provider**: in **Settings**, pick OpenAI / Anthropic / Gemini / Groq /
  OpenRouter and paste its API key. Keys are stored **server-side, encrypted at
  rest** (via `VFR_SECRET_KEY`), and never sent to the browser.

Since the backend runs on your own machine, local Ollama is a great no-cost
default here — no RAM ceiling to worry about.

---

## 10. Smoke test

Open your Vercel URL and walk through:

- [ ] `/login` renders the split-screen with any social buttons you enabled.
- [ ] **Create account** (email/password) → you land in the app, signed in.
- [ ] *(if enabled)* **Continue with Google / GitHub** completes and returns you signed in.
- [ ] **Ingest** a short `.txt` or paste text → it reports chunks added.
- [ ] **Search** returns results with source citations.
- [ ] **Ask** streams an answer (after [step 9](#9-choose-an-answer-model)).
- [ ] **Dashboard** shows *Your activity* counting up; **Reset my statistics** zeroes only your counters.
- [ ] **Sign out** returns to `/login`; protected routes redirect there when signed out.

---

## 11. Operational notes

**What's always up vs. on-demand.**

| Piece | Where | Always on? |
|-------|-------|-----------|
| Frontend | Vercel | ✅ yes |
| Accounts + stats | Neon Postgres | ✅ yes (survives your PC) |
| Backend API | your PC + tunnel | ⚠️ only while the process + tunnel run |
| Ingested docs / vector index | your PC (`indices/`) | persists on your disk between runs |

**Security — your backend is publicly reachable.** That's intended (the Vercel
frontend calls it), and it's gated:
- `VFR_AUTH__REQUIRED=true` makes the data endpoints reject anonymous requests.
- Passwords are bcrypt-hashed and never returned; JWTs are signed with your secret.
- The OAuth state cookie is auto-marked `Secure` because the tunnel URL is HTTPS.
- **Sign-up is open** by default (anyone can register an account). For a *private*
  demo, put the tunnel hostname behind **Cloudflare Access** (Zero Trust) — note
  that locks out the public Vercel frontend too unless you add a bypass, so it's
  for private/personal use, not a public portfolio link.

**No cold starts, but your PC must be awake.** Unlike a sleeping free dyno, your
backend responds instantly — as long as the machine is on and not asleep.
Disable sleep (or use `caffeinate -s` on macOS) if you want it reachable 24/7.

**Heavy ML stays local.** PyTorch, FAISS and the embedding model run on your
hardware, so there's no cloud RAM limit — the original reason for this setup.

**Reset statistics.** Each user resets only their own counters from the dashboard
(*Reset my statistics* → `POST /api/v1/auth/me/stats/reset`).

**Prefer a rented server instead?** This same backend runs on any host with
≥ 2 GB RAM (e.g. a Render *Standard* instance, a small VPS, or Fly.io): install
`requirements.txt`, set the same env vars, and start it with
`uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port $PORT`. The
Cloudflare Tunnel just replaces the need for that rented box.

---

## 12. Troubleshooting

**CORS error in the browser console** (`blocked by CORS policy`)
→ `VFR_API__CORS_ORIGINS` must be a JSON array containing your frontend origin
exactly (`https://…`, no trailing slash). Edit `.env`, restart the backend.

**`redirect_uri_mismatch` from Google / GitHub**
→ The provider's registered redirect URI must equal
`{VFR_AUTH__PUBLIC_BASE_URL}/api/v1/auth/{provider}/callback` character-for-character.

**`https://api.yourdomain.com` returns Cloudflare error 1033 / 502 / 530**
→ The tunnel isn't connected or can't reach the backend. Confirm
`cloudflared tunnel run vectorflow` is up, the backend is listening on
`127.0.0.1:8000`, and the `ingress` `service:` URL/port matches.

**API calls return a Cloudflare HTML challenge page instead of JSON**
→ Cloudflare is challenging the API subdomain. In the dashboard, lower the
security level for `api.yourdomain.com` (or add a WAF "skip" rule) and make sure
**Under Attack Mode** is off — challenges break `fetch`/XHR.

**Login works but every API call returns 401**
→ The frontend can't reach the backend or the token isn't attached. Confirm
`NEXT_PUBLIC_API_BASE_URL` points at the tunnel and the frontend was redeployed
after setting it. (A 401 also clears the stored token and bounces you to `/login`.)

**DB connection / SSL errors at boot**
→ Check `DATABASE_URL` and keep `?sslmode=require` for Neon. Confirm the Neon
project isn't paused (free projects idle out) and the region/host are correct.

**Questions never answer / time out**
→ No chat model configured. Complete [step 9](#9-choose-an-answer-model).

**Social buttons don't appear on `/login`**
→ That provider isn't configured. Verify both its client-ID and client-secret
are in `.env` and the backend was restarted; check `GET /api/v1/auth/providers`.
