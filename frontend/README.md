# VectorFlow-RAG Frontend

Next.js 14 (App Router) frontend for the VectorFlow-RAG backend. TypeScript +
Tailwind, a custom aurora design system (no UI kit), no global-state library, no
analytics, no cloud lock-in. Every page talks to the FastAPI backend over a
single typed HTTP contract.

## Quick start

```bash
cp .env.example .env.local           # set NEXT_PUBLIC_API_BASE_URL if non-default
npm install
npm run dev                          # → http://localhost:3000
```

The backend must be reachable at `NEXT_PUBLIC_API_BASE_URL` (default
`http://localhost:8000`). For a hosted deploy this is your tunnel URL (e.g. an
ngrok static domain). You can also override the backend at runtime, no rebuild,
from the browser console:

```js
localStorage.vfr_api_base_url = "https://your-backend"
```

## Scripts

```
npm run dev         # dev server (HMR)
npm run build       # production build
npm run start       # serve the build
npm run typecheck   # tsc --noEmit
npm run test        # vitest (component + unit) — 79 tests
npm run lint        # next lint
```

## Authentication & routing

The app is gated. `AppFrame` decides between full-bleed **auth screens** and the
authenticated app shell:

- **Auth routes** (`/login`, `/signup`, `/reset`, `/auth/callback`) render a
  Render-inspired split-screen with email/password + **Continue with Google /
  GitHub** (the social buttons appear only when the backend reports the provider
  configured).
- **Protected routes** redirect anonymous visitors to `/login?next=…`. A JWT is
  held in `localStorage`, attached as `Authorization: Bearer …` by the API
  client, and cleared automatically on any `401` (which bounces back to login).
- OAuth uses a top-level navigation to the backend; the callback returns the JWT
  in the URL fragment, which `/auth/callback` adopts.

When `VFR_AUTH__REQUIRED=false` (local default) the data endpoints stay open and
the gate is permissive; in production (`true`) login is required.

## Pages

| Route            | Purpose                                                            |
|------------------|-------------------------------------------------------------------|
| `/login` `/signup` `/reset` | Authentication (email/password + Google/GitHub OAuth)  |
| `/`              | Chat with streaming SSE answers + citations                       |
| `/search`        | Retrieval-only mode with full provenance                          |
| `/ingest`        | Drag-and-drop file upload + paste-text ingestion                  |
| `/documents`     | Indexed documents grouped by stable `doc_id`                      |
| `/dashboard`     | Per-user activity (with reset) + live process metrics + traces    |
| `/traces`        | Recent retrieval traces with expandable JSON detail               |
| `/settings`      | Pipeline status, **model providers**, **named indexes + benchmark**, jobs, cache, backend-URL override |

## Architecture

```
src/
├── app/                   # App Router pages (incl. login/signup/reset/auth/callback)
│   ├── layout.tsx         # aurora background + AuthProvider + AppFrame
│   └── <page>/page.tsx
├── components/
│   ├── ui/                # Button, Card, Input, Badge, Spinner, ErrorBox, StatCard, motion primitives
│   ├── brand/             # Logo / wordmark, Constellation canvas hero
│   ├── auth/              # AuthShell, AuthCard, ResetCard, SocialButtons, brand icons
│   ├── layout/            # AppFrame (auth gate + theme), Header (nav + UserMenu + ThemeToggle)
│   ├── chat/ · search/ · ingest/ · citations/
│   ├── dashboard/         # UserStatsPanel, MetricsPanel, TraceTable, DocumentsTable
│   └── settings/          # Providers/ModelsTab, IndexesTab, IndexBuilder, BenchmarkPanel, …
└── lib/
    ├── api/               # typed client (one module per resource)
    │   ├── client.ts      # fetch wrapper: base-URL resolution, bearer auth, ngrok-skip header, structured errors
    │   ├── sse.ts         # SSE parser for fetch-based streams
    │   ├── types.ts       # TS types mirroring backend Pydantic schemas
    │   └── auth.ts · indexes.ts · jobs.ts · models.ts · …
    ├── auth/              # token store + AuthContext (status, user, login/signup/logout, OAuth adopt)
    ├── hooks/             # useApi, usePolling, useStreamingAsk, useJobProgress, useTheme
    └── utils/             # format helpers, cn
```

The frontend never imports backend code and never assumes a colocated backend.
The only contract is the typed schemas in `src/lib/api/types.ts`.

## Deployment (Vercel)

1. Import the repo and **set the Root Directory to `frontend`** (the app lives in
   this subfolder).
2. Framework auto-detects as Next.js; leave build/output at defaults
   (`vercel.json` pins the framework + adds baseline security headers).
3. Set `NEXT_PUBLIC_API_BASE_URL` to your backend tunnel URL. It is inlined at
   **build time**, so changing it later requires a redeploy.

| Scenario                    | How it runs                                                   |
|-----------------------------|---------------------------------------------------------------|
| Local dev                   | `npm run dev` + Python backend on localhost                   |
| Hosted (free)               | Vercel build + backend on your Mac via a free tunnel (ngrok)  |
| Desktop (Tauri / Electron)  | `OUTPUT=export npm run build`, ship `./out` + Python sidecar   |

Full deployment runbook: [`../DEPLOYMENT.md`](../DEPLOYMENT.md).

## Internationalization & rendering

Unicode-safe and direction-aware without UI string translation:

- **Bidirectional text**: user-content containers (chat answers, source text,
  document names) use `dir="auto"`, so the browser picks LTR/RTL per the first
  strong character — Arabic/Hebrew render correctly with no language branching.
- **Font stack** resolves to the OS UI font (broad script coverage: CJK, Arabic,
  Hebrew) — no web-font download, consistent with local-first operation. Display
  type uses Space Grotesk / Inter / JetBrains Mono via `next/font`.
- **Locale formatting**: numbers/dates use `toLocaleString()`.
- **UI string i18n** is intentionally out of scope — Unicode-safe *rendering*
  matters more than translated chrome; `next-intl` can be layered on later.

## Tests

Vitest + React Testing Library on happy-dom. Tests live in `tests/`:

* `tests/unit/`        — API client, SSE parser, format helpers, hooks
* `tests/components/`  — AuthCard, SourcePanel, DropZone, MetricsPanel, Indexes/Models tabs, Bidi, Settings

```
npm run test           # 79 tests pass
```
</content>
