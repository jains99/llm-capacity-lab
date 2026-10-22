# Capacity Lab

**Live demo:** <https://jains99.github.io/llm-capacity-lab/>

An interactive, single-page simulator for LLM inference capacity. It shows how model weights, KV cache, context length, concurrency, memory bandwidth, prefill compute, and cache hit rates decide whether a workload fits on one GPU or needs several.

Everything runs in the browser. There is no API, database, or server-side component, so the built site can be hosted on any static host.

## The five lenses

One shared configuration drives five views:

- **Overview** — the overall verdict, throughput, cost, and bottlenecks.
- **Memory** — weights, KV cache, overhead, free VRAM, and the per-GPU budget.
- **Latency** — TTFT, TPOT, queueing, and the effect of batching.
- **Caching** — exact, semantic, and shared-prefix cache hits; self-hosted and hosted-API prompt caching.
- **Topology** — replicas versus tensor parallelism, interconnect cost, and multi-GPU layout.

The left rail controls the model, GPU count, context, users, quantization, cache rates, and targets. The URL always contains the current configuration, so a link reproduces the exact scenario.

## Run locally

Requires Node.js 20.19+ or 22.12+.

```bash
npm install
npm run dev
```

Open the URL printed by Vite, normally <http://localhost:5173/>.

Or use the managed helper:

```bash
./start.sh          # starts Vite, waits for readiness, opens the browser
PORT=5180 ./start.sh
NO_OPEN=1 ./start.sh
./stop.sh
```

`start.sh` writes `.capacity-lab.pid` and `.capacity-lab.log`; `stop.sh` stops the managed process.

## Build and test

```bash
npm run build       # type-check and produce dist/
npm run preview     # serve the production build locally
npm test            # formula and catalogue regression tests
npm run lint
```

The app is a static bundle. `dist/` can be uploaded to any static web server.

## Hosting

### GitHub Pages (included)

GitHub Pages works well for this project. It is a static site and does not need SPA server rewrites; all state lives in the query string.

The repository includes [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml), which tests, builds, and deploys on every push to `main` or `master`.

Setup:

1. Push the project to a GitHub repository.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Push to `main` (or `master`), or run **Deploy to GitHub Pages** manually from the Actions tab.

The workflow gets the Pages base path from `actions/configure-pages` and passes it to Vite as `VITE_BASE_PATH`. Project sites are served from `https://<user>.github.io/<repo>/`, so assets must use that prefix.

If you use a **user/organization site** (`<user>.github.io`) or a **custom domain**, the base path should be `/`. `configure-pages` normally handles this. If the deployed page loads without CSS or JavaScript, add a repository variable:

- **Settings → Secrets and variables → Actions → Variables → New repository variable**
- Name: `VITE_BASE_PATH`
- Value: `/`

For free GitHub accounts, Pages is available for public repositories. Private-repository Pages requires a paid GitHub plan.

### Vercel

Import the repository at <https://vercel.com/new>. Vercel detects Vite automatically; the included `vercel.json` also sets:

- Build command: `npm run build`
- Output directory: `dist`

No environment variables are needed.

### Netlify

Import the repository at <https://app.netlify.com/start>, or connect it with the Netlify CLI. The included `netlify.toml` sets:

- Build command: `npm run build`
- Publish directory: `dist`
- Node version: 22

No environment variables are needed.

### Cloudflare Pages

Create a Pages project from the repository and use:

- Build command: `npm run build`
- Build output directory: `dist`

### Any static server

After `npm run build`, upload `dist/` to S3, Cloudflare R2, nginx, Caddy, GitHub Pages, or any other static host. Serve `index.html` at the site root. If you deploy under a subpath, build with that path:

```bash
VITE_BASE_PATH=/my-subpath/ npm run build
```

## Notes and limits

- The simulator is a planning model, not a benchmark. It uses roofline-style estimates, an Erlang-C queue, and published/representative hardware figures. Real throughput depends on the runtime, kernels, batching policy, and interconnect.
- The model catalogue includes dense, MoE, and hybrid-attention models. **Qwen3.8 27B** uses 16 full-attention KV layers out of 64 total layers; the KV and prefill-attention math accounts for that.
- The GPU catalogue is restricted to cards with at least 12 GB of VRAM.
- WebGL is required for the 3D stage. The app shows a fallback message if WebGL is unavailable.
- First-time visitors get a short onboarding tour. It can be replayed with **How it works** in the header.
- Onboarding completion is stored in `localStorage` under `capacity-lab:onboarding-v1`.
- Reduced-motion preferences and keyboard navigation are supported.

## License

MIT © 2026 Sachin Jain. See [LICENSE](LICENSE).
