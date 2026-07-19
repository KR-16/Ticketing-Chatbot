/* ============================================================
   Project Showcase — client app
   Hash router: #/ (catalog) and #/p/<slug> (product page)
   ============================================================ */

const SITE = window.SITE_CONFIG || {
  owner_name: "Your Name",
  handle: "yourhandle",
  intro: "Selected projects, presented as products.",
  github_url: "https://github.com/yourhandle",
};

async function loadProjects() {
  if (Array.isArray(window.PROJECTS_DATA)) return window.PROJECTS_DATA;
  try {
    const res = await fetch("./data/projects.json", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    return await res.json();
  } catch (e) {
    return null;
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function inlineMd(s) {
  s = esc(s);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function mdToHtml(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "", list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (let raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }
    let m;
    if ((m = line.match(/^###\s+(.*)/))) { closeList(); html += `<h3>${inlineMd(m[1])}</h3>`; continue; }
    if ((m = line.match(/^##\s+(.*)/)))  { closeList(); html += `<h3>${inlineMd(m[1])}</h3>`; continue; }
    if ((m = line.match(/^[-*]\s+(.*)/))) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inlineMd(m[1])}</li>`; continue;
    }
    if ((m = line.match(/^\d+\.\s+(.*)/))) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inlineMd(m[1])}</li>`; continue;
    }
    closeList();
    html += `<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return html;
}

const app = () => document.getElementById("app");
const statusLabel = { live: "Live", beta: "Beta", experimental: "Experimental", archived: "Archived" };
function statusPill(s) {
  const key = (s || "experimental").toLowerCase();
  return `<span class="status ${key}"><span class="dot"></span>${statusLabel[key] || s}</span>`;
}
function ghIcon() {
  return `<svg class="ico" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38v-1.34c-2.23.49-2.7-1.07-2.7-1.07-.36-.93-.89-1.18-.89-1.18-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.71 1.23 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.83-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.22 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.52.56.83 1.28.83 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>`;
}
function extIcon() {
  return `<svg class="ico" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M6 3H3.5A1.5 1.5 0 0 0 2 4.5v8A1.5 1.5 0 0 0 3.5 14h8a1.5 1.5 0 0 0 1.5-1.5V10M9 2h5v5M14 2 7 9"/></svg>`;
}

function renderCatalog(projects) {
  const ordered = [...projects].sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
  const cards = ordered.map((p) => {
    const tags = (p.tech_stack || []).slice(0, 4)
      .map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    const lang = p.primary_language ? `<span class="kv"><b>${esc(p.primary_language)}</b></span>` : "";
    const lic = p.license ? `<span class="kv">${esc(p.license)}</span>` : "";
    return `
      <a class="card view-link" href="#/p/${encodeURIComponent(p.slug)}">
        <div class="card-top">
          <h3>${esc(p.name)}</h3>
          ${statusPill(p.status)}
        </div>
        <p class="tagline">${esc(p.tagline || "")}</p>
        <div class="tags">${tags}</div>
        <div class="spec-row">${lang}${lic}
          ${p.year ? `<span class="kv">${esc(p.year)}</span>` : ""}
        </div>
      </a>`;
  }).join("");

  return `
    <div class="view">
      <header class="topbar"><div class="wrap">
        <div class="brand">${esc(SITE.owner_name)}<span class="tick">/</span><span class="brand-sub">projects</span></div>
        <nav><a href="${esc(SITE.github_url)}" target="_blank" rel="noopener">github ↗</a></nav>
      </div></header>

      <section class="hero"><div class="wrap">
        <span class="eyebrow kicker">Project catalog · @${esc(SITE.handle)}</span>
        <h1>Things I've built, shipped as <em>products</em>.</h1>
        <p>${esc(SITE.intro)}</p>
        <div class="hero-meta">
          <span><b>${projects.length}</b> project${projects.length === 1 ? "" : "s"}</span>
          <span>Each with a full spec sheet</span>
          <span>Source on <b>GitHub</b></span>
        </div>
      </div></section>

      <div class="wrap">
        <div class="catalog-head">
          <span class="eyebrow">Index</span>
          <span class="count">${projects.length} entr${projects.length === 1 ? "y" : "ies"}</span>
        </div>
        <div class="grid">${cards}</div>
      </div>

      ${footer()}
    </div>`;
}

function renderProduct(p) {
  const features = (p.features || []).map((f, i) => `
    <div class="feature">
      <div class="idx">${String(i + 1).padStart(2, "0")}</div>
      <div>
        <h4>${esc(f.title)}</h4>
        ${f.detail ? `<p>${esc(f.detail)}</p>` : ""}
      </div>
    </div>`).join("");

  const shots = (p.screenshots || []).map((s) => `
    <figure class="shot">
      <img src="${esc(s.src)}" alt="${esc(s.alt || p.name)}" loading="lazy">
      ${s.caption ? `<figcaption>${esc(s.caption)}</figcaption>` : ""}
    </figure>`).join("");

  const ctas = [];
  if (p.demo_url) ctas.push(`<a class="btn primary" href="${esc(p.demo_url)}" target="_blank" rel="noopener">${extIcon()} Live demo</a>`);
  if (p.repo_url) ctas.push(`<a class="btn" href="${esc(p.repo_url)}" target="_blank" rel="noopener">${ghIcon()} View source</a>`);
  if (p.docs_url) ctas.push(`<a class="btn" href="${esc(p.docs_url)}" target="_blank" rel="noopener">${extIcon()} Docs</a>`);

  const dsRow = (label, val) => val ? `<div class="ds-row"><dt>${label}</dt><dd>${val}</dd></div>` : "";
  const chips = (arr) => `<div class="chips">${(arr || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>`;

  const datasheet = `
    <aside class="datasheet">
      <div class="ds-head"><span class="t">Spec sheet</span><span class="n">${esc(p.slug)}</span></div>
      <dl>
        ${dsRow("Status", `${statusLabel[(p.status||"").toLowerCase()] || p.status || "—"}`)}
        ${dsRow("Language", p.primary_language ? esc(p.primary_language) : "")}
        ${dsRow("Stack", (p.tech_stack && p.tech_stack.length) ? chips(p.tech_stack) : "")}
        ${dsRow("License", p.license ? esc(p.license) : "")}
        ${dsRow("Topics", (p.topics && p.topics.length) ? chips(p.topics) : "")}
        ${dsRow("Updated", p.updated ? esc(p.updated) : "")}
        ${dsRow("Repo", p.repo_url ? `<a href="${esc(p.repo_url)}" target="_blank" rel="noopener">${esc((p.repo_url||"").replace(/^https?:\/\/(www\.)?github\.com\//, ""))}</a>` : "")}
      </dl>
    </aside>`;

  return `
    <div class="view">
      <header class="topbar"><div class="wrap">
        <div class="brand">${esc(SITE.owner_name)}<span class="tick">/</span><span class="brand-sub">projects</span></div>
        <nav><a href="#/">← all projects</a></nav>
      </div></header>

      <div class="wrap">
        <a class="back" href="#/"><span class="arrow">←</span> Back to catalog</a>

        <section class="product-hero">
          <div class="row1">${statusPill(p.status)}
            ${p.primary_language ? `<span class="eyebrow">${esc(p.primary_language)}</span>` : ""}
          </div>
          <h1>${esc(p.name)}</h1>
          <p class="lede">${esc(p.tagline || "")}</p>
          <div class="cta-row">${ctas.join("")}</div>
        </section>

        ${shots ? `<div class="shot-band">${shots}</div>` : ""}

        <div class="product-body">
          <div class="main">
            ${p.problem ? `<div class="section"><span class="eyebrow">The problem</span><div class="prose">${mdToHtml(p.problem)}</div></div>` : ""}
            <div class="section"><span class="eyebrow">Overview</span><div class="prose">${mdToHtml(p.overview || "_No overview yet._")}</div></div>
            ${features ? `<div class="section"><span class="eyebrow">Capabilities</span><h2>What it does</h2><div class="features">${features}</div></div>` : ""}
          </div>
          ${datasheet}
        </div>
      </div>

      ${footer()}
    </div>`;
}

function footer() {
  return `
    <footer class="site-foot"><div class="wrap" style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;width:100%;">
      <span class="note">Built by ${esc(SITE.owner_name)} · updated continuously</span>
      <div class="links"><a href="${esc(SITE.github_url)}" target="_blank" rel="noopener">GitHub ↗</a><a href="#/">Catalog</a></div>
    </div></footer>`;
}

function renderError() {
  app().innerHTML = `<div class="wrap"><div class="empty">
    <h2>Couldn't load projects</h2>
    <p>Expected <code>./data/projects.json</code> next to this page.</p>
  </div></div>`;
}
function renderEmpty() {
  app().innerHTML = `<div class="wrap"><div class="empty">
    <h2>No projects yet</h2>
    <p>Run the <code>project-showcase</code> skill on a repo to add the first entry.</p>
  </div></div>`;
}
function renderNotFound() {
  app().innerHTML = `<div class="wrap"><div class="empty">
    <h2>Project not found</h2>
    <p><a href="#/">← Back to the catalog</a></p>
  </div></div>`;
}

let PROJECTS = null;

function route() {
  if (PROJECTS === null) return renderError();
  if (PROJECTS.length === 0) return renderEmpty();

  const hash = location.hash || "#/";
  const m = hash.match(/^#\/p\/(.+)$/);
  if (m) {
    const slug = decodeURIComponent(m[1]);
    const p = PROJECTS.find((x) => x.slug === slug);
    if (!p) return renderNotFound();
    app().innerHTML = renderProduct(p);
  } else {
    app().innerHTML = renderCatalog(PROJECTS);
  }
  window.scrollTo({ top: 0 });
}

async function main() {
  PROJECTS = await loadProjects();
  window.addEventListener("hashchange", route);
  route();
}
main();
