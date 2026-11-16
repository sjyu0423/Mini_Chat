# W6 release review — Mini_Chat website

Date: 2026-11-16
Branch: `v1.0-release` @ `8101752960c54a9c5c1bcf986b68812259d8fcf1`
Target URL: `https://sjyu0423.github.io/Mini_Chat/`
Vite base: `/Mini_Chat/`

No commit, push, Pages enablement, or workflow run was performed.

---

## 1. Git status

- Branch: `v1.0-release`, up to date with `origin/v1.0-release`
- Modified tracked: `.gitignore` (adds `.claude/`)
- Untracked: entire `website/`, `.github/workflows/deploy-website.yml`
- `.claude/` present locally and now ignored by root `.gitignore`
- ML code, checkpoints, datasets, and experiment reports untouched

## 2. Files changed (W6)

### Added
- `.github/workflows/deploy-website.yml` — manual Pages deploy
- `website/src/styles/fonts.css` — latin + latin-ext only
- `website/public/og-image.png`, `robots.txt`, `sitemap.xml`
- `website/docs/w6_release_review.md`

### Updated
- `website/index.html` — title, description, canonical, Open Graph, Twitter
- `website/src/config/site.ts` — `pagesUrl`
- `website/src/main.tsx` — use latin font CSS
- `website/public/favicon.svg` — brand-aligned colors
- `website/.gitignore` — stronger excludes
- Root `.gitignore` — ignore `.claude/`
- Disclosure / source link wrapping (Under the hood, Historical aside, scaling, charts, samples)
- Explicit `focus-visible` on major `<summary>` controls
- `website/README.md` — deployment + data sources + checklist
- Roadmap copy: website “ready for GitHub Pages” (still in-progress until published)

### Removed
- `website/src/sections/home/PlannedSections.tsx` (no-op stub)
- Unused duplicate fixtures `bpeExample.json`, `experiments/curves.json`

## 3. UI polish and accessibility

- Fixed genuine 320px horizontal overflow from unbroken mono documentation paths
- Kept dark steel-blue + IBM Plex; no redesign
- Charts retain captions / `aria-label`; scaling table retains `<caption>` + `scope`
- Recorded demo: native `<select>` + checkbox; recorded / no-API labeling unchanged
- Skip link and global `:focus-visible` outline retained
- **Keyboard / Under the hood:** summary receives focus; click toggles open. Enter/Space via available browser automation did **not** toggle native `<details>` (tooling limitation — DOM/key helpers are not trusted activation). Native HTML disclosure is used; real OS keyboard should work in a normal browser, but this session cannot claim a trusted-key pass.

## 4. Performance and bundle

Production build (`npm run check` passed):

| Asset | Size (approx.) |
|---|---|
| JS | ~313 kB (~96 kB gzip) |
| CSS | ~32 kB (~7 kB gzip) |
| Fonts | latin + latin-ext + mono only (Cyrillic/Greek/Vietnamese subsets removed) |
| OG image | ~44 kB PNG |

- No large new runtime dependencies
- Bundle mentions `serve_chat` only as documentation copy; no `/api/chat` string
- Only `fetch(` present is Vite/modulepreload helper, not inference

## 5. SEO and metadata

- Title: `Mini_Chat — GPT trained from scratch`
- Meta description states static / no live model
- Canonical + `og:*` + Twitter card → `https://sjyu0423.github.io/Mini_Chat/`
- `og-image.png` (1200×630) branded card
- `robots.txt` allows indexing; `sitemap.xml` lists the homepage
- Favicon SVG aligned to design tokens

## 6. Deployment workflow

File: `.github/workflows/deploy-website.yml`

- `on: workflow_dispatch` only
- Job `build`: Node 22, `npm ci`, `npm run check`, upload `website/dist`
- Job `deploy`: `actions/deploy-pages@v4` with `github-pages` environment
- Minimal permissions: `contents:read`, `pages:write`, `id-token:write`
- Does not run other ML workflows (none existed previously)

## 7. Verified target URL and base path

- Remote: `https://github.com/sjyu0423/Mini_Chat.git`
- Intended Pages URL: `https://sjyu0423.github.io/Mini_Chat/`
- `vite.config.ts` `base: '/Mini_Chat/'` matches repository name
- Preview verified under `/Mini_Chat/` locally
- GitHub API Pages status could not be queried here (`gh` Forbidden); treat Pages as **not yet confirmed enabled**

## 8. Browser-test results

Production preview under `/Mini_Chat/`:

| Check | Result |
|---|---|
| 1440 / 768 / 390 / 320 overflow | Pass after path-wrapping fix |
| Nav anchors + section ids | Present (`overview`, `architecture`, `experiments`, `engineering`, `engineering-demo`) |
| Architecture + Under the hood (click) | Pass |
| Historical M1 aside | Present |
| Experiments timeline / charts / scaling | Present |
| Engineering modules + recorded demo select/compare | Pass (prior + this session) |
| SEO assets (favicon, og, robots, sitemap) | HTTP 200 on preview |
| Inference API requests | None observed |
| Reduced-motion CSS | Present in stylesheet |
| Trusted keyboard toggle of `<details>` | **Not confirmed** (tooling) |

Dark-only site: no light-mode test applicable.

## 9. Git tracking and ignore audit

**Include (proposed):**
- `website/` source, config, lockfile, public assets, `docs/w6_release_review.md`
- `.github/workflows/deploy-website.yml`
- Updated root `.gitignore`

**Exclude (verified ignored):**
- `website/node_modules/`, `website/dist/`, `*.tsbuildinfo`, `.vite/`
- `.env*` (except optional `.env.example` if added later)
- `.claude/`
- `/runs/`, `/data/`, checkpoints (root ignore)

## 10. Exact commit / staging recommendation

Do **not** run until explicitly approved:

```bash
git add .gitignore
git add .github/workflows/deploy-website.yml
git add website
git status   # confirm: no .claude, no node_modules, no dist, no runs/
```

Suggested commit message (when approved):

```
Add Mini_Chat static website and manual GitHub Pages workflow.

Ship the completed portfolio site under website/ with evidence-backed
content and a workflow_dispatch deploy path for GitHub Pages.
```

Do not `git add .` from the repo root (would risk scanning unrelated untracked paths before ignore evaluation in interactive review). Prefer the explicit paths above.

## 11. Remaining blockers / manual setup

1. Your approval to stage + commit + push
2. Enable GitHub Pages with **GitHub Actions** as the source
3. Manually run **Deploy website**
4. Optional: confirm Pages URL after first deploy; update portfolio link if desired
5. Trusted keyboard audit of disclosures on a physical keyboard (tooling gap above)

## 12. Final publication checklist

- [ ] Approve W6 review
- [ ] Stage recommended paths only
- [ ] Commit on `v1.0-release`
- [ ] Push to `origin`
- [ ] Settings → Pages → GitHub Actions
- [ ] Actions → Deploy website → Run workflow
- [ ] Open `https://sjyu0423.github.io/Mini_Chat/`
- [ ] Spot-check recorded demo + nav on phone
- [ ] Confirm Network tab has no `/api/chat`

## Content / evidence note

`docs/showcase/facts.json` contains **25** catalogued claims. Key public numbers on the site (13,989,888; 29,676,544; 8,192 / 7,931; 157,286,400; EOT ~97.5%; bpb 1.264 / 1.204; 16,813 SFT chats; 74 ms; MPS 2.2→13.6 GB; tests 305+20; fingerprint `e6be77cc…`) were re-checked against that catalog and site copy. No metrics or recorded replies were invented or rewritten in W6.
