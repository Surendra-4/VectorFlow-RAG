# VectorFlow-RAG Frontend

Next.js 14 App Router frontend for the VectorFlow-RAG backend. TypeScript +
Tailwind, no global state library, no UI kit, no analytics, no cloud
dependencies. All pages talk to the FastAPI backend over typed HTTP.

## Quick start

```bash
cp .env.example .env.local           # set NEXT_PUBLIC_API_BASE_URL if non-default
npm install
npm run dev                          # → http://localhost:3000
```

Backend must be reachable at `NEXT_PUBLIC_API_BASE_URL` (default
`http://localhost:8000`). The frontend's Settings page also lets users
override the backend URL at runtime via localStorage.

## Scripts

```
npm run dev         # dev server (HMR)
npm run build       # production build
npm run start       # serve the build
npm run typecheck   # tsc --noEmit
npm run test        # vitest (component + unit)
npm run test:watch  # vitest in watch mode
npm run lint        # next lint
```

## Static export (for desktop packaging)

```
OUTPUT=export npm run build
# produces ./out — serveable from any static host or sidecar process
```

## Pages

| Route          | Purpose                                                       |
|----------------|---------------------------------------------------------------|
| `/`            | Chat with streaming SSE answers + citations                   |
| `/search`      | Retrieval-only mode with full provenance                      |
| `/ingest`      | Drag-and-drop file upload + paste-text ingestion              |
| `/documents`   | Indexed documents grouped by stable `doc_id`                  |
| `/dashboard`   | Live metrics (counters, latencies, cache, request breakdown)  |
| `/traces`      | Recent retrieval traces with expandable JSON detail           |
| `/settings`    | Pipeline status, cache controls, backend-URL override         |

## Architecture

```
src/
├── app/                   # Next.js App Router pages
│   ├── layout.tsx         # root layout, theme, header, footer
│   ├── error.tsx          # global error boundary
│   └── <page>/page.tsx
├── components/
│   ├── ui/                # Button, Card, Input, Badge, Spinner, ErrorBox
│   ├── layout/            # Header, StatusBadge
│   ├── chat/              # ChatInterface, MessageBubble, LatencyChip
│   ├── ingest/            # IngestForm, DropZone, IngestSummary
│   ├── search/            # SearchForm
│   ├── citations/         # SourcePanel
│   └── dashboard/         # MetricsPanel, TraceTable, DocumentsTable, SettingsPanel
└── lib/
    ├── api/               # typed API client (one module per resource)
    │   ├── client.ts      # fetch wrapper with structured errors
    │   ├── errors.ts      # ApiError class
    │   ├── sse.ts         # SSE parser for fetch-based streams
    │   ├── types.ts       # TS types mirroring backend Pydantic schemas
    │   └── <resource>.ts
    ├── hooks/             # useApi, usePolling, useStreamingAsk, useTheme
    └── utils/             # format helpers, cn
```

## Deployment topology

| Scenario                         | How it runs                                                          |
|----------------------------------|----------------------------------------------------------------------|
| Local dev                        | `npm run dev` + Python backend on localhost                          |
| Local production                 | `npm run build && npm run start` + Python backend on localhost       |
| Vercel (hosted lightweight)      | Vercel build + user-supplied backend URL                             |
| Desktop (Tauri / Electron)       | `OUTPUT=export npm run build`, ship `./out` + Python sidecar         |

The frontend never imports backend code and never assumes a colocated
backend. The only contract is the typed schemas in `src/lib/api/types.ts`.

## Internationalization & rendering (Phase 11)

The frontend is Unicode-safe and direction-aware without UI string translation:

- **Bidirectional text**: user-content containers (chat answers, source text,
  document names) use `dir="auto"`, so the browser picks LTR/RTL per the first
  strong character. Arabic/Hebrew content renders correctly with no
  language-specific branching.
- **Font fallback stack** (in `tailwind.config.ts`):
  `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`.
  This resolves to the OS's native UI font, which carries broad script
  coverage: PingFang/Hiragino (macOS CJK), Microsoft YaHei/Meiryo (Windows
  CJK), Noto (Linux), and system Arabic/Hebrew faces. No web-font download
  is required — consistent with local-first/offline operation.
- **Locale formatting**: numbers/dates use `toLocaleString()` (browser locale).
- **UI string translation (i18n)** is intentionally out of scope for this
  phase — Unicode-safe *rendering* matters far more than translated chrome,
  and `next-intl` can be layered on later without structural change.

## Tests

Vitest + React Testing Library on happy-dom. Tests live in `tests/`:

* `tests/unit/`        — API client, SSE parser, format helpers, hooks
* `tests/components/`  — SourcePanel, DropZone, MetricsPanel

```
npm run test           # 58 tests pass
```
