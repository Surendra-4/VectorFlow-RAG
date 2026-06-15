# Deploying VectorFlow-RAG — the ₹0 setup

This app loads PyTorch, sentence-transformers, FAISS and a local embedding model
(~1 GB RAM) — too much for free cloud servers. So we **run the backend on your
own Mac**, expose it through a **free tunnel**, host the lightweight **frontend
on Vercel free tier**, and keep accounts/stats in a **free Postgres tier**.

**No paid servers, no paid domain, no credit card, ₹0 recurring.**

- **Frontend** → [Vercel](https://vercel.com) free tier (Next.js)
- **Backend** → your Mac (FastAPI), published via a **free tunnel** (ngrok or Cloudflare)
- **Database** → free **Postgres** ([Neon](https://neon.tech) / [Supabase](https://supabase.com)) — or even local SQLite

> Your backend is online only while your Mac is on and the tunnel is running.
> The frontend (Vercel) and database stay up regardless; the API answers when
> your machine does.

---

## Does it work behind a tunnel URL that changes between sessions?

**Short answer: the app works fine behind a changing URL — except OAuth sign-in
and the public frontend link, which want a *stable* hostname.** Here's exactly
what does and doesn't care:

| Component | Needs a stable URL? | Why |
|-----------|:---:|-----|
| Email/password sign-up & login (JWT) | **No** | The JWT is signed/verified with `VFR_AUTH__JWT_SECRET` — independent of hostname. |
| CORS | **No** | It allow-lists your **Vercel** origin (stable). The backend's own URL isn't a CORS origin. |
| OAuth *state* cookie / HTTPS cookie flag | **No** | Set and read on the backend host within a single login; both tunnels are HTTPS. |
| Ingest / search / ask (incl. streaming) | **No** | They just need to reach the backend, wherever it is. |
| **Frontend → backend** (`NEXT_PUBLIC_API_BASE_URL`) | **Yes** | Baked into the Vercel build. New URL ⇒ update the env var + redeploy (or use the `localStorage` override below). |
| **Backend** `VFR_AUTH__PUBLIC_BASE_URL` | **Yes** | Used to build OAuth callbacks. New URL ⇒ edit `.env` + restart. |
| **OAuth redirect URIs** (Google/GitHub) | **Yes** | The provider rejects any callback that doesn't match what you registered. New URL ⇒ re-register in the provider console. |

So nothing is *architecturally* broken by an ephemeral URL — email/password +
all RAG features work. The friction is purely **OAuth** (re-register each
session) and **re-pointing the Vercel frontend** each session.

**The free fix for a permanent hostname (no domain purchase): an ngrok free
static domain.** The free ngrok plan gives every account **one static domain**
like `https://your-name.ngrok-free.app`, with **no credit card**. It never
changes between sessions — so OAuth and the Vercel config become one-time setups.
That's **Option A** below, and it's what I recommend.

---

## Choose your tunnel

| | **Option A — ngrok static domain** (recommended) | **Option B — Cloudflare Quick Tunnel** |
|---|---|---|
| Public URL | **Stable** `https://your-name.ngrok-free.app` | **Ephemeral** `https://<random>.trycloudflare.com` (new each run) |
| Account | Free ngrok account (no card) | **None** |
| OAuth (Google/GitHub) | Register **once** | Re-register **every session** |
| Vercel frontend | Set `NEXT_PUBLIC_API_BASE_URL` **once** | Re-point + redeploy each session (or `localStorage` override) |
| Caveat | One-time "browser warning" interstitial — already handled in-app (see below) | None, but the URL churns |

Both are ₹0. Pick **A** for a shareable, set-and-forget link; **B** for a
throwaway personal demo with zero signup.

---

## Before you start

Free accounts/tools (none require a credit card):

- [ ] [GitHub](https://github.com) — the repo (Vercel deploys from it).
- [ ] [Vercel](https://vercel.com) — frontend hosting (free Hobby).
- [ ] [Neon](https://neon.tech) **or** [Supabase](https://supabase.com) — free Postgres. *(Or skip and use local SQLite.)*
- [ ] **Option A:** an [ngrok](https://ngrok.com) account (free Personal plan) + `brew install ngrok`.
- [ ] **Option B:** `brew install cloudflared` (no account).
- [ ] Python deps: from the repo root, `pip install -r requirements.txt`.
- [ ] *(optional)* Google / GitHub OAuth apps — only practical with Option A.
- [ ] *(optional, free)* [Ollama](https://ollama.com) for a local answer model.

---

## 1. Database (free)

**Neon:** create a project → create a database `vectorflow` → copy the
connection string (keep `?sslmode=require`):
```
postgresql://user:password@ep-xxx-123.us-east-2.aws.neon.tech/vectorflow?sslmode=require
```
The app normalizes the driver to `psycopg2` and creates its tables on first boot.

> **Zero-account alternative:** leave `DATABASE_URL` unset and the app uses a
> local SQLite file (`./var/app.db`) on your Mac — durable, no signup. Since the
> backend is single-instance on your machine, this is perfectly fine.

---

## 2. Configure & run the backend on your Mac

```bash
cp .env.production.example .env
```
Edit `.env` and set:
- `DATABASE_URL` — from step 1 (or remove the line for SQLite).
- `VFR_AUTH__JWT_SECRET` — `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `VFR_SECRET_KEY` — `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `VFR_AUTH__PUBLIC_BASE_URL` — your tunnel URL (fill after step 3 the first time).
- `VFR_AUTH__FRONTEND_URL` — your Vercel URL (fill after step 4).
- `VFR_API__CORS_ORIGINS` — `["https://<your-vercel-app>.vercel.app"]`.

Start it (binds to `127.0.0.1:8000`, auto-loads `.env`):
```bash
python -m src.api
```
A healthy boot ends with `Uvicorn running on http://127.0.0.1:8000`. Check:
`curl http://127.0.0.1:8000/health`.

---

## 3. Start the tunnel

### Option A — ngrok static domain (stable)

```bash
brew install ngrok
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN     # from dashboard.ngrok.com ▸ Your Authtoken
```
Claim your free static domain at **dashboard.ngrok.com ▸ Domains** (you get one,
e.g. `your-name.ngrok-free.app`). Then run, with the backend still up:
```bash
ngrok http 8000 --url=https://your-name.ngrok-free.app
#  older ngrok:  ngrok http 8000 --domain=your-name.ngrok-free.app
```
Or use the committed config: copy `deploy/ngrok/ngrok.example.yml` to
`~/.config/ngrok/ngrok.yml`, edit the two values, and run `ngrok start vectorflow`.

Set `VFR_AUTH__PUBLIC_BASE_URL=https://your-name.ngrok-free.app` in `.env` and
restart the backend. Verify: `https://your-name.ngrok-free.app/health`.

> **ngrok's browser-warning page** is already handled: the frontend sends the
> `ngrok-skip-browser-warning` header on every request, so the API returns real
> JSON/SSE, not the interstitial. Nothing to configure.

### Option B — Cloudflare Quick Tunnel (ephemeral, zero-account)

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```
It prints `https://<random-words>.trycloudflare.com`. Use that as your tunnel
URL — but it **changes every run**, so follow the
[per-session checklist](#6-per-session-checklist-option-b-only) each time.

---

## 4. Deploy the frontend on Vercel

1. `git push origin main` (if not already pushed).
2. Vercel → **Add New ▸ Project** → import this repo.
3. **Set Root Directory to `frontend`** (the app lives in that subfolder).
4. Add an Environment Variable (Production):

   | Key | Value |
   |-----|-------|
   | `NEXT_PUBLIC_API_BASE_URL` | your tunnel URL (e.g. `https://your-name.ngrok-free.app`) |

5. Deploy → you get `https://<project>.vercel.app`.
6. Put that Vercel URL into the backend `.env` as `VFR_AUTH__FRONTEND_URL` and
   `VFR_API__CORS_ORIGINS`, then restart the backend.

> `NEXT_PUBLIC_*` is baked at build time — changing it later needs a redeploy
> (Deployments ▸ ⋯ ▸ Redeploy).

---

## 5. Environment variables — what changes, and when

| Variable | Where | ngrok static (Option A) | Quick tunnel (Option B) |
|----------|-------|----|----|
| `DATABASE_URL` | backend `.env` | once | once |
| `VFR_AUTH__JWT_SECRET` | backend `.env` | once | once |
| `VFR_SECRET_KEY` | backend `.env` | once | once |
| `VFR_AUTH__FRONTEND_URL` | backend `.env` | once (Vercel URL) | once |
| `VFR_API__CORS_ORIGINS` | backend `.env` | once (Vercel URL) | once |
| `VFR_AUTH__PUBLIC_BASE_URL` | backend `.env` | once | **every session** |
| `NEXT_PUBLIC_API_BASE_URL` | Vercel | once | **every session** (+ redeploy) |
| OAuth client id/secret | backend `.env` | once | re-register each session |

---

## 6. Per-session checklist (Option B only)

With the ngrok static domain (Option A) there's **nothing** to do each session —
just start the backend and `ngrok start vectorflow`. With Quick Tunnel, each time
the URL changes:

1. Copy the new `https://<random>.trycloudflare.com`.
2. Backend: set `VFR_AUTH__PUBLIC_BASE_URL` to it in `.env`, restart `python -m src.api`.
3. Frontend: update `NEXT_PUBLIC_API_BASE_URL` on Vercel and redeploy — **or**,
   for a personal demo, skip the rebuild and run in the browser console:
   ```js
   localStorage.vfr_api_base_url = "https://<random>.trycloudflare.com"
   ```
4. If you use OAuth: update the redirect URIs in the Google/GitHub consoles.

---

## 7. Enable Google & GitHub sign-in (optional)

Email/password works without this. OAuth is practical **only with a stable URL
(Option A)** — the callback must match `VFR_AUTH__PUBLIC_BASE_URL` exactly.

- **Google** ▸ [Cloud Console](https://console.cloud.google.com) → OAuth client (Web) →
  Authorized redirect URI: `https://your-name.ngrok-free.app/api/v1/auth/google/callback`.
- **GitHub** ▸ [Developer settings ▸ OAuth Apps](https://github.com/settings/developers) →
  callback URL: `https://your-name.ngrok-free.app/api/v1/auth/github/callback`.

Put the client id/secret in `.env` (`VFR_AUTH__GOOGLE_*` / `VFR_AUTH__GITHUB_*`),
restart the backend. The buttons then appear on `/login`.

---

## 8. Choose an answer model

Search works out of the box; **answering questions needs a chat model**. Since
the backend is on your Mac, **local Ollama is the free, no-key choice**:
```bash
brew install ollama && ollama pull llama3.2
```
It serves at `http://localhost:11434` next to the backend — pick the model in
**Settings**. (Or paste a hosted-provider API key in Settings; keys are stored
server-side, encrypted with `VFR_SECRET_KEY`, never sent to the browser.)

---

## 9. Smoke test

Open your Vercel URL and confirm: create an account → ingest a short `.txt` →
search returns cited results → ask streams an answer → the dashboard's *Your
activity* counts up → *Reset my statistics* zeroes only your counters → sign out
redirects to `/login`.

---

## 10. Operational notes

- **Keep it running:** the API is reachable only while `python -m src.api` and
  the tunnel are up. Keep them in a `tmux`/terminal, and stop your Mac from
  sleeping (`caffeinate -s python -m src.api`) for 24/7 reach.
- **ngrok free limits:** one online tunnel and a monthly bandwidth cap — plenty
  for a portfolio demo. The static domain and HTTPS are included.
- **Security:** your backend is public but gated — `VFR_AUTH__REQUIRED=true`
  rejects anonymous data calls, passwords are bcrypt-hashed, JWTs are signed.
  Sign-up is open by default (anyone can register); that's normal for a demo.
- **Data durability:** accounts + stats live in Postgres (or local SQLite);
  ingested documents live in `indices/` on your Mac and persist between runs.
- **Prefer always-on later?** The same backend runs on any ≥ 2 GB host with
  `uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port $PORT` — the
  tunnel just removes the need to rent one.

---

## 11. Troubleshooting

**API returns an ngrok HTML "You are about to visit…" page instead of JSON**
→ The `ngrok-skip-browser-warning` header handles this in-app; if you call the
API from `curl`/Postman, add `-H "ngrok-skip-browser-warning: 1"`.

**CORS error in the console** → `VFR_API__CORS_ORIGINS` must be a JSON array with
your exact Vercel origin (no trailing slash). Edit `.env`, restart the backend.

**`redirect_uri_mismatch`** → The provider's registered redirect URI must equal
`{VFR_AUTH__PUBLIC_BASE_URL}/api/v1/auth/{provider}/callback` exactly. With Quick
Tunnel this breaks whenever the URL changes — use Option A for OAuth.

**Login works but API calls 401** → `NEXT_PUBLIC_API_BASE_URL` is wrong or the
frontend wasn't redeployed after changing it; or the tunnel/backend is down.

**`trycloudflare.com` / ngrok URL unreachable** → the tunnel process or the
backend (`127.0.0.1:8000`) isn't running, or the tunnel printed a new URL you
haven't propagated yet ([§6](#6-per-session-checklist-option-b-only)).

**DB connection / SSL errors at boot** → check `DATABASE_URL`, keep
`?sslmode=require`, and confirm the Neon project isn't paused (free projects idle
out and resume on the next connection).

**Questions never answer** → no chat model configured; see [step 8](#8-choose-an-answer-model).
