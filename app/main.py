"""HTTP surface: the Bazarr webhook, a manual trigger, and a status page."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse)
from pydantic import BaseModel

from . import auth as auth_mod
from . import sources
from . import ui
from .bazarr import BazarrClient
from .config import settings
from .endpoints import Pool, parse as parse_endpoints
from .store import Store
from .worker import Sweeper, Worker, display_title, series_key

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("tarjem")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store(settings.data_dir / "tarjem.db")
    bazarr = BazarrClient(settings)

    # One worker per backend by default: routing alone buys nothing, since a
    # single worker sends one batch at a time whichever machine answers it.
    pool = Pool(_seed_backends(store))
    # Headroom for a backend added from the UI later: a worker with nothing
    # free simply waits, so a spare costs nothing.
    count = max(settings.workers, len(pool.endpoints) + 1) if pool else max(1, settings.workers)
    if pool:
        log.info("%d backend(s) in the pool: %s", len(pool.endpoints),
                 ", ".join(e.name for e in pool.endpoints))

    workers = [Worker(settings, store, bazarr, name=f"worker-{i + 1}", pool=pool)
               for i in range(count)]
    # A lane of its own for anything picked by hand. Whatever provider it names,
    # the point is that it starts now instead of behind a backlog the sweeper
    # built. A local rush does share the GPU with the main worker, so the two
    # interleave - still far better than waiting out the queue.
    workers.append(Worker(settings, store, bazarr, name="rush",
                          priority_only=True, pool=pool))
    for worker in workers:
        worker.start()

    sweeper = Sweeper(settings, store, bazarr)
    if settings.sweep_enabled:
        sweeper.start()

    state.update(store=store, bazarr=bazarr, workers=workers, sweeper=sweeper,
                 pool=pool, started=time.time())
    log.info("tarjem up | provider=%s model=%s target=%s register=%s",
             settings.provider, settings.active_model, settings.target_lang, settings.register)
    auth_mod.warn_if_open(settings)
    if not bazarr.ping():
        log.warning("Bazarr not reachable at %s - webhook still works, sweeping will not",
                    settings.bazarr_url)
    try:
        yield
    finally:
        for worker in workers:
            worker.stop()
        sweeper.stop()
        bazarr.close()


app = FastAPI(title="tarjem", description="AI Arabic subtitles for the *arr stack", lifespan=lifespan)


auth = auth_mod.make_dependency(settings)


@app.get("/login", response_class=HTMLResponse)
def login_form(error: str = Query(default="")) -> str:
    return ui.LOGIN.replace(
        "__ERROR__", "<div class='err' style='margin-top:12px;font-size:12px'>"
                     "Wrong password.</div>" if error else "")


@app.post("/login")
def login(request: Request, password: str = Form(default="")) -> Response:
    if not auth_mod.password_ok(settings, password):
        log.warning("failed sign-in attempt from %s",
                    request.headers.get("cf-connecting-ip")
                    or (request.client.host if request.client else "?"))
        return RedirectResponse("/login?error=1", status_code=303)

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth_mod.COOKIE,
        auth_mod.issue(settings),
        max_age=settings.session_hours * 3600,
        httponly=True,          # not readable from JavaScript
        samesite="lax",         # not sent on cross-site form posts
        # Decided per request, not per deployment. Forcing it on would mean a
        # session obtained over plain http on the LAN is never sent back, so
        # signing in there would appear to succeed and then silently fail.
        secure=settings.cookie_secure or auth_mod.is_https(request),
        path="/",
    )
    return response


@app.post("/logout")
def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth_mod.COOKIE, path="/")
    return response


def store() -> Store:
    return state["store"]


def pool() -> Pool:
    return state["pool"]


def _seed_backends(db: Store) -> list:
    """Backends live in the database so the UI can change them.

    LLM_ENDPOINTS only seeds an empty table - after that the database wins,
    otherwise a redeploy would silently undo every change made in the UI.
    """
    stored = db.backends()
    if not stored and settings.llm_endpoints:
        for endpoint in parse_endpoints(settings.llm_endpoints):
            db.put_backend(endpoint.name, endpoint.kind, endpoint.url, endpoint.model)
            log.info("seeded backend %s from LLM_ENDPOINTS", endpoint.name)
        stored = db.backends()

    out = []
    for row in stored:
        (endpoint,) = parse_endpoints(f"{row['kind']}@{row['url']}#{row['model']}") or (None,)
        if endpoint is None:
            log.warning("stored backend %s is malformed, skipping", row["name"])
            continue
        endpoint.name, endpoint.enabled = row["name"], row["enabled"]
        out.append(endpoint)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    video: str
    subtitle: str | None = None
    title: str | None = None
    force: bool = False
    # Run this one job somewhere other than the configured default - e.g.
    # "anthropic" for an episode you want tonight rather than in two hours.
    provider: str | None = None
    rush: bool = False


KNOWN_PROVIDERS = ("anthropic", "openai", "openai-compatible", "ollama", "local")


def _check_provider(name: str | None) -> str:
    """Reject an unusable provider now, rather than after the job is queued."""
    if not name:
        return ""
    name = name.strip().lower()
    if name not in KNOWN_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider '{name}' - one of {', '.join(KNOWN_PROVIDERS)}",
        )
    if name == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY is not set, so the anthropic provider cannot run",
        )
    if name in ("openai", "openai-compatible") and not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
    return name


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health(request: Request) -> dict:
    """Deliberately reachable without credentials - the container healthcheck
    and any uptime monitor need it - but it only says "ok" to a stranger."""
    if not auth_mod.authenticated(settings, request):
        return {"status": "ok"}
    return {
        "status": "ok",
        "uptime_s": round(time.time() - state["started"]),
        "provider": settings.provider,
        "model": settings.active_model,
        "target_lang": settings.target_lang,
        "register": settings.register,
        "bazarr": state["bazarr"].ping(),
        "jobs": store().counts(),
        "dry_run": settings.dry_run,
        "auth": settings.auth_enabled,
        "workers": len(state["workers"]),
        "backends": state["pool"].status() if state["pool"] else [],
    }


async def _payload(request: Request) -> dict:
    """Bazarr's post-processing command runs through `sh -c`.

    A JSON body survives that only if no title contains an apostrophe, so the
    documented command posts urlencoded form fields instead. Both shapes are
    accepted here.
    """
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="malformed JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        return body
    form = await request.form()
    return {k: v for k, v in form.items()}


def _int(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


@app.post("/hook/bazarr", dependencies=[Depends(auth)])
async def bazarr_hook(request: Request) -> JSONResponse:
    """Called by Bazarr's custom post-processing when it downloads a subtitle.

    Bazarr fires this for *every* download, including the Arabic file this
    service just produced, so the target language is filtered out here rather
    than relying on the command in Bazarr's settings being written carefully.
    """
    data = await _payload(request)
    if not data.get("video"):
        raise HTTPException(status_code=400, detail="missing 'video'")

    lang = str(data.get("lang") or "").split(":")[0].strip().lower()
    if lang and lang in sources.aliases(settings.target_lang):
        return JSONResponse({"queued": False, "reason": f"{lang} is the target language"})

    video = Path(settings.to_local(str(data["video"])))
    if not video.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {video}")

    existing = sources.has_target(video, settings)
    if existing:
        return JSONResponse({"queued": False, "reason": f"already have {existing.name}"})

    episode_id = _int(data.get("episode_id"))
    series_id = _int(data.get("series_id"))
    movie_id = _int(data.get("movie_id"))
    kind = "episode" if episode_id else ("movie" if movie_id else "")

    job_id = store().enqueue(
        str(video), display_title(video), series_key(video), "bazarr",
        kind, series_id, episode_id or movie_id,
    )
    if job_id is None:
        return JSONResponse({"queued": False, "reason": "already queued"})
    if data.get("subtitle"):
        store().update(job_id, detail=str(data["subtitle"]))
    log.info("queued job %s from Bazarr hook: %s (source lang %s)", job_id, video.name, lang or "?")
    return JSONResponse({"queued": True, "job": job_id})


@app.post("/translate", dependencies=[Depends(auth)])
def translate(req: TranslateRequest) -> JSONResponse:
    video = Path(settings.to_local(req.video))
    if not video.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {video}")

    existing = sources.has_target(video, settings)
    if existing and not req.force:
        return JSONResponse(
            {"queued": False, "reason": f"already have {existing.name}; pass force=true to redo"}
        )
    if existing and req.force:
        existing.rename(existing.with_suffix(existing.suffix + ".bak"))

    provider = _check_provider(req.provider)
    job_id = store().enqueue(
        str(video), req.title or display_title(video), series_key(video),
        "rush" if req.rush else "manual",
        provider=provider, priority=1 if req.rush else 0,
    )
    if job_id is None:
        return JSONResponse({"queued": False, "reason": "already queued"})
    if req.subtitle:
        store().update(job_id, detail=req.subtitle)
    return JSONResponse({
        "queued": True, "job": job_id,
        "provider": provider or settings.provider, "rush": req.rush,
    })


@app.post("/jobs/{job_id}/rush", dependencies=[Depends(auth)])
def rush(job_id: int, provider: str = Query(default="anthropic")) -> dict:
    """Re-run a job now, on a faster provider, ahead of the queue."""
    name = _check_provider(provider)
    new_id = store().requeue(job_id, provider=name, priority=1)
    if new_id is None:
        raise HTTPException(status_code=404, detail="no such job, or it is already queued")
    return {"queued": True, "job": new_id, "provider": name}


@app.post("/sweep", dependencies=[Depends(auth)])
def sweep(limit: int = Query(default=0, ge=0, le=1000)) -> dict:
    state.pop("library", None)          # the queue just changed; re-read it
    return state["sweeper"].sweep(limit or None)


LIBRARY_TTL = 300


def _library(refresh: bool = False) -> tuple[list[dict], str]:
    """The whole library, each item marked with what tarjem knows about it.

    Walking MEDIA_ROOTS and asking Bazarr both cost real time, so a page
    refresh should not redo them.
    """
    cached = state.get("library")
    if cached and not refresh and time.time() - cached["at"] < LIBRARY_TTL:
        return cached["items"], cached["source"]

    videos, wanted = state["sweeper"].everything()
    db = store()
    items = []
    for cand in videos:
        video = cand["video"]
        path = str(video)
        existing = sources.has_target(video, settings)
        items.append({
            "video": path,
            "title": cand["title"],
            "series": cand["key"],
            "name": video.name,
            "translated": existing is not None,
            "subtitle": existing.name if existing else "",
            "pending": db.is_pending(path),
            # Bazarr still wants the target language for this one.
            "wanted": path in wanted,
        })
    items.sort(key=lambda i: (i["series"].lower(), i["name"].lower()))
    source = "disk + bazarr" if wanted else "disk"
    state["library"] = {"items": items, "source": source, "at": time.time()}
    return items, source


@app.get("/api/library", dependencies=[Depends(auth)])
def api_library(
    q: str = Query(default=""),
    state_filter: str = Query(default="all", alias="state"),
    limit: int = Query(default=5000, ge=1, le=20000),
    refresh: bool = Query(default=False),
) -> dict:
    items, source = _library(refresh)
    total_all = len(items)

    if q:
        needle = q.casefold()
        items = [i for i in items if needle in i["title"].casefold()
                 or needle in i["name"].casefold()]
    if state_filter == "missing":
        items = [i for i in items if not i["translated"]]
    elif state_filter == "translated":
        items = [i for i in items if i["translated"]]

    return {
        "source": source,
        "library_total": total_all,
        "total": len(items),
        "items": items[:limit],
    }


@app.get("/jobs", dependencies=[Depends(auth)])
def jobs(limit: int = Query(default=50, ge=1, le=500), status: str | None = None) -> dict:
    return {"jobs": store().recent(limit, status, running_first=True),
            "counts": store().counts()}


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def job(job_id: int) -> dict:
    found = store().get(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="no such job")
    return found


@app.post("/jobs/{job_id}/retry", dependencies=[Depends(auth)])
def retry(job_id: int) -> dict:
    new_id = store().requeue(job_id)
    if new_id is None:
        raise HTTPException(status_code=404, detail="no such job, or it is already queued")
    return {"requeued": True, "job": new_id}


class BackendRequest(BaseModel):
    kind: str = "ollama"
    url: str
    model: str = ""


@app.get("/api/backends", dependencies=[Depends(auth)])
def api_backends() -> dict:
    return {"backends": pool().status()}


@app.post("/api/backends", dependencies=[Depends(auth)])
def add_backend(req: BackendRequest) -> dict:
    spec = f"{req.kind.strip().lower()}@{req.url.strip()}#{req.model.strip()}"
    parsed = parse_endpoints(spec)
    if not parsed:
        raise HTTPException(status_code=400,
                            detail="need a kind and a url, e.g. ollama@http://host:11434")
    endpoint = parsed[0]
    if endpoint.kind not in ("ollama", "local", "openai", "openai-compatible",
                             "lmstudio", "anthropic"):
        raise HTTPException(status_code=400, detail=f"unknown kind '{endpoint.kind}'")
    if pool().find(endpoint.name):
        raise HTTPException(status_code=409, detail=f"{endpoint.name} is already configured")

    store().put_backend(endpoint.name, endpoint.kind, endpoint.url, endpoint.model)
    pool().add(endpoint)
    log.info("added backend %s", endpoint.name)
    return {"added": endpoint.name, "backends": pool().status()}


@app.patch("/api/backends/{name}", dependencies=[Depends(auth)])
def toggle_backend(name: str, enabled: bool = Query(...)) -> dict:
    """Turn a backend off without losing its settings.

    A job already running on it is left to finish - stopping mid-file would
    throw away everything translated so far.
    """
    if not pool().set_enabled(name, enabled):
        raise HTTPException(status_code=404, detail=f"no backend named {name}")
    store().set_backend_enabled(name, enabled)
    return {"name": name, "enabled": enabled, "backends": pool().status()}


@app.delete("/api/backends/{name}", dependencies=[Depends(auth)])
def delete_backend(name: str) -> dict:
    endpoint = pool().find(name)
    if endpoint is None:
        raise HTTPException(status_code=404, detail=f"no backend named {name}")
    if endpoint.lock.locked():
        raise HTTPException(
            status_code=409,
            detail="that backend is mid-job. Disable it instead; it will stop "
                   "taking new work and you can remove it once it is idle.",
        )
    pool().remove(name)
    store().drop_backend(name)
    return {"removed": name, "backends": pool().status()}


@app.get("/glossaries", dependencies=[Depends(auth)])
def glossaries() -> dict:
    return {"glossaries": store().glossaries()}


@app.get("/glossaries/{key}", dependencies=[Depends(auth)])
def glossary(key: str) -> dict:
    found = store().get_glossary(key)
    if not found:
        raise HTTPException(status_code=404, detail="no glossary for that key")
    return found


@app.delete("/glossaries/{key}", dependencies=[Depends(auth)])
def drop_glossary(key: str) -> dict:
    store().drop_glossary(key)
    return {"deleted": key}


@app.get("/jobs/{job_id}/subtitle", response_class=PlainTextResponse, dependencies=[Depends(auth)])
def subtitle(job_id: int) -> str:
    found = store().get(job_id)
    if not found or not found.get("output"):
        raise HTTPException(status_code=404, detail="no output for that job")
    path = Path(found["output"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="output file is gone")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Status page
# --------------------------------------------------------------------------

def _footer() -> str:
    """The rail's small print: what is doing the work right now."""
    backends = pool().status() if pool() else []
    if backends:
        live = sum(1 for b in backends if b["enabled"] and b["healthy"])
        detail = f"{live}/{len(backends)} backends up"
    else:
        detail = f"{settings.provider} &middot; {settings.active_model}"
    return (f"{detail}<br>{settings.target_lang} &middot; {settings.register}"
            f"<br>bazarr {'ok' if state['bazarr'].ping() else 'unreachable'}")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, status: str = ""):
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)

    db = store()
    counts = db.counts()
    pills = "".join(
        f"<a class='pill {k}' href='/?status={k}'>{k} {v}</a>"
        for k, v in sorted(counts.items())
    ) or "<span class='pill'>no jobs yet</span>"

    # The default view leaves the queue out: with a real backlog it is dozens
    # of identical rows that bury both the running job and the history.
    rows_source = (db.recent(60, status=status, running_first=True) if status
                   else db.recent(40, running_first=True, exclude=("queued",)))

    rows = []
    for j in rows_source:
        pct = int((j.get("progress") or 0) * 100)
        stats = j.get("stats") or {}
        if j["status"] == "done" and stats:
            result = (f"{stats.get('translated', 0)}/{stats.get('total_cues', 0)} cues"
                      f" &middot; {stats.get('seconds', 0)}s")
        elif j["status"] == "failed":
            result = f"<span class='err'>{_esc(j.get('error', ''))}</span>"
        else:
            result = _esc(j.get("stage", ""))

        action = ""
        if j["status"] in ("failed", "done", "skipped"):
            action = (f"<button onclick=\"rushJob(this,{j['id']},'ollama')\">local</button>"
                      f"<button onclick=\"rushJob(this,{j['id']},'anthropic')\">Claude</button>")
        badge = " <span class='pill'>rush</span>" if j.get("priority") else ""
        where = _esc(j.get("provider") or j.get("origin") or "")

        rows.append(
            f"<tr><td class='sub'>{j['id']}</td>"
            f"<td>{_esc(j.get('title') or Path(j['video']).name)}{badge}"
            f"<br><span class='sub'>{_esc(Path(j['video']).name)}</span></td>"
            f"<td><span class='pill {j['status']}'>{j['status']}</span></td>"
            f"<td><span class='bar'><i style='width:{pct}%'></i></span>"
            f"<span class='sub'>{pct}%</span></td>"
            f"<td class='sub'>{where}</td><td>{result}</td>"
            f"<td style='white-space:nowrap'>{action}</td></tr>"
        )

    keyhint = ("" if settings.anthropic_api_key else
               " <span class='warn'>ANTHROPIC_API_KEY is not set.</span>")
    if settings.api_token:
        keyhint += " <span class='sub'>Open with ?token=… for the buttons to work.</span>"

    body = f"""
<div class='toolbar'>{pills}
  {"<a class='pill' href='/'>show all activity</a>" if status
   else "<span class='sub'>queued rows hidden &middot; click a count to filter</span>"}
</div>

<div class='panel'>
  <h2>Translate now</h2>
  <div class='inner'>
    <input id='path' style='width:min(560px,60vw)' spellcheck='false'
           placeholder='/media/tv/Show/Season 1/Episode.mkv'>
    <button class='primary' onclick="rushPath('ollama')">Local</button>
    <button onclick="rushPath('anthropic')">Claude</button>
    <span id='msg' class='msg'></span>
    <div class='sub' style='margin-top:9px'>Jumps the queue and runs in its own
      lane.{keyhint}</div>
  </div>
</div>

<div class='panel'>
  <h2>Activity</h2>
  <table>
    <thead><tr><th>#</th><th>Title</th><th>Status</th><th>Progress</th>
      <th>Where</th><th>Result</th><th></th></tr></thead>
    <tbody>{"".join(rows) or "<tr><td colspan='7' class='empty'>Nothing yet</td></tr>"}</tbody>
  </table>
</div>"""

    script = ui.JS_BASE + """
async function rushPath(provider) {
  const v = document.getElementById("path").value.trim();
  if (!v) return say("paste a path first", false);
  say("queueing…", true);
  try {
    const d = await api("POST", "/translate",
                        {video: v, provider, rush: true, force: true});
    say(d.queued ? `queued as job ${d.job}` : d.reason, d.queued);
  } catch (e) { say(e.message, false); }
}
async function rushJob(btn, id, provider) {
  btn.disabled = true; say("queueing…", true);
  try {
    const d = await api("POST", `/jobs/${id}/rush?provider=${provider}`);
    say(`queued as job ${d.job} on ${provider}`, true);
  } catch (e) { btn.disabled = false; say(e.message, false); }
}
"""
    return ui.shell(title="tarjem", active="jobs", heading="Activity",
                    body=body, script=script, footer=_footer(), refresh=15)


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request):
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)

    body = """
<div class='toolbar'>
  <input id='q' style='width:min(340px,45vw)' placeholder='Filter by title or filename'
         oninput='render()'>
  <button class='tab on' id='t-all' onclick="setFilter('all')">All</button>
  <button class='tab' id='t-missing' onclick="setFilter('missing')">Missing</button>
  <button class='tab' id='t-translated' onclick="setFilter('translated')">Translated</button>
  <button onclick='load(true)'>Rescan</button>
  <span id='msg' class='msg'></span>
</div>
<div class='sub' id='counts' style='margin-bottom:14px'>Reading the library&hellip;</div>

<div class='panel'>
  <table>
    <thead><tr><th>Show / Film</th><th>File</th><th>Subtitle</th><th></th></tr></thead>
    <tbody id='rows'><tr><td colspan='4' class='empty'>Loading&hellip;</td></tr></tbody>
  </table>
</div>"""

    script = ui.JS_BASE + """
let ITEMS = [], FILTER = "all", SOURCE = "";
function setFilter(f) {
  FILTER = f;
  for (const t of ["all", "missing", "translated"])
    document.getElementById("t-" + t).classList.toggle("on", t === f);
  render();
}
async function load(refresh) {
  say(refresh ? "rescanning…" : "loading…", true);
  try {
    const d = await api("GET", "/api/library" + (refresh ? "?refresh=true" : ""));
    ITEMS = d.items; SOURCE = d.source; say("", true); render();
  } catch (e) { say(e.message, false); }
}
function render() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  let rows = ITEMS.filter(i => !q || i.title.toLowerCase().includes(q)
                                  || i.name.toLowerCase().includes(q));
  if (FILTER === "missing") rows = rows.filter(i => !i.translated);
  if (FILTER === "translated") rows = rows.filter(i => i.translated);

  const done = ITEMS.filter(i => i.translated).length;
  document.getElementById("counts").innerHTML =
    ITEMS.length + " videos &middot; " + done + " translated &middot; " +
    (ITEMS.length - done) + " without a subtitle &middot; via " + esc(SOURCE);

  document.getElementById("rows").innerHTML = rows.slice(0, 1500).map(function (i) {
    const state = i.translated
        ? "<span class='pill done'>" + esc(i.subtitle) + "</span>"
        : i.pending ? "<span class='pill running'>queued</span>"
                    : "<span class='pill skipped'>none</span>";
    const dis = i.pending ? "disabled" : "";
    const redo = i.translated ? " (redo)" : "";
    const v = encodeURIComponent(i.video);
    return "<tr><td>" + esc(i.title) + "</td>" +
      "<td class='sub mono'>" + esc(i.name) + "</td><td>" + state + "</td>" +
      "<td style='white-space:nowrap'>" +
      "<button class='primary' " + dis + " onclick=\\"go(this,'" + v + "','ollama'," +
        (!!i.translated) + ")\\">Local" + redo + "</button>" +
      "<button " + dis + " onclick=\\"go(this,'" + v + "','anthropic'," +
        (!!i.translated) + ")\\">Claude" + redo + "</button>" +
      "</td></tr>";
  }).join("") || "<tr><td colspan='4' class='empty'>Nothing matches</td></tr>";
}
async function go(btn, video, provider, translated) {
  const path = decodeURIComponent(video);
  if (translated && !confirm(
      "This already has a subtitle.\\n\\nRe-translate it? The existing file is " +
      "renamed to .bak, not deleted.")) return;
  btn.disabled = true; say("queueing…", true);
  try {
    const d = await api("POST", "/translate",
                        {video: path, force: !!translated, provider: provider, rush: true});
    say(d.queued ? "queued as job " + d.job : d.reason, d.queued);
  } catch (e) { btn.disabled = false; say(e.message, false); }
}
load(false);
"""
    return ui.shell(title="tarjem library", active="library", heading="Library",
                    body=body, script=script, footer=_footer())


@app.get("/backends", response_class=HTMLResponse)
def backends_page(request: Request):
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)

    body = """
<div class='sub' style='margin-bottom:14px'>Machines that do the translating.
  Disable one to get its GPU back &mdash; a job already running on it finishes
  first.</div>

<div class='panel'>
  <h2>Backends</h2>
  <table>
    <thead><tr><th>Backend</th><th>Model</th><th>State</th><th></th></tr></thead>
    <tbody id='rows'><tr><td colspan='4' class='empty'>Loading&hellip;</td></tr></tbody>
  </table>
</div>

<div class='panel'>
  <h2>Add a backend</h2>
  <div class='inner'>
    <select id='kind'>
      <option value='ollama'>Ollama</option>
      <option value='openai'>OpenAI-compatible (LM Studio, vLLM&hellip;)</option>
      <option value='anthropic'>Anthropic</option>
    </select>
    <input id='url' style='width:min(300px,45vw)' spellcheck='false'
           placeholder='http://192.168.1.50:11434'>
    <input id='model' style='width:min(220px,35vw)' spellcheck='false'
           placeholder='command-r7b-arabic'>
    <button class='primary' onclick='add()'>Add</button>
    <span id='msg' class='msg'></span>
    <div class='sub' style='margin-top:9px'>An Ollama backend uses the native
      API, which enforces the output schema. Anything else goes through the
      OpenAI-compatible path.</div>
  </div>
</div>"""

    script = ui.JS_BASE + """
function render(list) {
  document.getElementById("rows").innerHTML = list.map(function (b) {
    let state = b.enabled ? "<span class='pill enabled'>enabled</span>"
                          : "<span class='pill skipped'>disabled</span>";
    if (b.enabled && !b.healthy)
      state += " <span class='pill failed'>unreachable, retry in " +
               b.down_for_s + "s</span>";
    if (b.busy) state += " <span class='pill running'>translating</span>";
    return "<tr><td>" + esc(b.name) + "<br><span class='sub'>" + esc(b.kind) +
      "</span></td>" +
      "<td class='sub mono'>" + esc(b.model || "(default)") + "</td>" +
      "<td>" + state + "</td>" +
      "<td style='white-space:nowrap'>" +
      "<button onclick=\\"toggle('" + esc(b.name) + "'," + (!b.enabled) + ")\\">" +
        (b.enabled ? "Disable" : "Enable") + "</button>" +
      "<button class='danger' " + (b.busy ? "disabled" : "") +
        " onclick=\\"remove('" + esc(b.name) + "')\\">Remove</button>" +
      "</td></tr>";
  }).join("") || "<tr><td colspan='4' class='empty'>No backends configured</td></tr>";
}
async function load() {
  try { render((await api("GET", "/api/backends")).backends); }
  catch (e) { say(e.message, false); }
}
async function toggle(name, enabled) {
  try {
    const d = await api("PATCH",
      "/api/backends/" + encodeURIComponent(name) + "?enabled=" + enabled);
    render(d.backends);
    say(name + (enabled ? " enabled" : " disabled"), true);
  } catch (e) { say(e.message, false); }
}
async function remove(name) {
  if (!confirm("Remove " + name +
               "? Its settings are lost - disable instead to keep them.")) return;
  try {
    const d = await api("DELETE", "/api/backends/" + encodeURIComponent(name));
    render(d.backends); say(name + " removed", true);
  } catch (e) { say(e.message, false); }
}
async function add() {
  const url = document.getElementById("url").value.trim();
  if (!url) return say("a url is required", false);
  try {
    const d = await api("POST", "/api/backends", {
      kind: document.getElementById("kind").value, url: url,
      model: document.getElementById("model").value.trim()});
    render(d.backends); say("added " + d.added, true);
    document.getElementById("url").value = "";
  } catch (e) { say(e.message, false); }
}
load();
setInterval(load, 10000);
"""
    return ui.shell(title="tarjem backends", active="backends", heading="Backends",
                    body=body, script=script, footer=_footer())


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
