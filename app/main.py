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


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>tarjem</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#14161a;
      color:#d8dee9;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
 form{{background:#1b1f26;border:1px solid #262a31;border-radius:8px;padding:28px;width:300px}}
 h1{{font-size:18px;margin:0 0 4px}} p{{color:#7b8494;font-size:12px;margin:0 0 18px}}
 input{{font:inherit;width:100%;box-sizing:border-box;background:#14161a;color:#d8dee9;
       border:1px solid #3b4252;border-radius:4px;padding:8px}}
 button{{font:inherit;width:100%;margin-top:12px;background:#5e81ac;color:#eceff4;border:0;
        border-radius:4px;padding:9px;cursor:pointer}}
 button:hover{{background:#81a1c1}}
 .err{{color:#bf616a;font-size:12px;margin-top:12px}}
</style></head><body>
<form method="post" action="/login">
  <h1>tarjem</h1>
  <p>AI Arabic subtitles</p>
  <input type="password" name="password" placeholder="password" autofocus
         autocomplete="current-password">
  <button type="submit">Sign in</button>
  {error}
</form></body></html>"""


@app.get("/login", response_class=HTMLResponse)
def login_form(error: str = Query(default="")) -> str:
    return LOGIN_PAGE.format(
        error='<div class="err">Wrong password.</div>' if error else ""
    )


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

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>tarjem</title><meta http-equiv="refresh" content="10">
<style>
 body{{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#14161a;color:#d8dee9;
      margin:0;padding:24px}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#7b8494;margin-bottom:20px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #262a31;vertical-align:top}}
 th{{color:#7b8494;font-weight:500}}
 .done{{color:#a3be8c}} .failed{{color:#bf616a}} .running{{color:#ebcb8b}}
 .queued{{color:#81a1c1}} .skipped{{color:#6b7280}}
 .bar{{display:inline-block;height:6px;background:#2e3440;width:90px;border-radius:3px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#88c0d0}}
 .pill{{background:#22262d;padding:2px 8px;border-radius:10px;margin-right:6px;
        text-decoration:none;display:inline-block}}
 a.pill:hover{{background:#3b4252}}
 .path{{color:#8fbcbb}} .err{{color:#bf616a;font-size:12px}}
 button{{font:inherit;font-size:12px;background:#3b4252;color:#e5e9f0;border:1px solid #4c566a;
        border-radius:4px;padding:3px 8px;cursor:pointer}}
 button:hover{{background:#4c566a}} button:disabled{{opacity:.5;cursor:default}}
 input{{font:inherit;font-size:13px;background:#1b1f26;color:#d8dee9;border:1px solid #3b4252;
       border-radius:4px;padding:5px 8px;width:520px;max-width:60vw}}
 .rushbar{{margin:0 0 18px;padding:12px;background:#1b1f26;border:1px solid #262a31;border-radius:6px}}
 .hint{{color:#7b8494;font-size:12px;margin-top:6px}}
 #msg{{margin-left:10px;font-size:12px}}
</style></head><body>
<h1>tarjem <a href="/library" style="font-size:12px">library</a> <a href="/backends" style="font-size:12px">backends</a> {logout}</h1>
<div class="sub">{provider} &middot; {model} &middot; {target}/{register} &middot; bazarr {bazarr}</div>
<div style="margin-bottom:16px">{pills}</div>

<div class="rushbar">
  <input id="path" placeholder="/media/tv/Show/Season 1/Episode.mkv"
         value="" spellcheck="false">
  <button onclick="rushPath('ollama')">Translate now (local)</button>
  <button onclick="rushPath('anthropic')">Translate now (Claude)</button>
  <span id="msg"></span>
  <div class="hint">Either one jumps the queue and runs in its own lane rather
    than waiting behind the backlog.{keyhint}</div>
</div>

<table><tr><th>#</th><th>title</th><th>status</th><th>progress</th><th>source</th><th>result</th><th></th></tr>
{rows}
</table>

<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const hdrs = TOKEN ? {{"x-api-token": TOKEN, "Content-Type": "application/json"}}
                   : {{"Content-Type": "application/json"}};
function say(t, ok) {{
  const m = document.getElementById("msg");
  m.textContent = t;
  m.style.color = ok ? "#a3be8c" : "#bf616a";
}}
async function post(url, body) {{
  const r = await fetch(url, {{method: "POST", headers: hdrs,
                              body: body ? JSON.stringify(body) : null}});
  const d = await r.json().catch(() => ({{}}));
  if (!r.ok) throw new Error(d.detail || r.status);
  return d;
}}
async function rushPath(provider) {{
  const v = document.getElementById("path").value.trim();
  if (!v) return say("paste a path first", false);
  say("queueing...", true);
  try {{
    const d = await post("/translate",
                         {{video: v, provider: provider, rush: true, force: true}});
    say(d.queued ? `queued as job ${{d.job}}` : d.reason, d.queued);
  }} catch (e) {{ say(String(e.message), false); }}
}}
async function rushJob(btn, id, provider) {{
  btn.disabled = true;
  say("queueing...", true);
  try {{
    const d = await post(`/jobs/${{id}}/rush?provider=${{provider}}`);
    say(`queued as job ${{d.job}} on ${{provider}}`, true);
  }} catch (e) {{ btn.disabled = false; say(String(e.message), false); }}
}}
</script>
</body></html>"""


LIBRARY_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>tarjem library</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#14161a;
      color:#d8dee9;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} a{color:#88c0d0}
 .sub{color:#7b8494;margin-bottom:18px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #262a31}
 th{color:#7b8494;font-weight:500;position:sticky;top:0;background:#14161a}
 .path{color:#8fbcbb;font-size:12px}
 .done{color:#a3be8c} .pending{color:#ebcb8b} .missing{color:#7b8494}
 button{font:inherit;font-size:12px;background:#3b4252;color:#e5e9f0;border:1px solid #4c566a;
        border-radius:4px;padding:3px 8px;cursor:pointer;margin-right:4px}
 button:hover{background:#4c566a} button:disabled{opacity:.45;cursor:default}
 button.claude{border-color:#5e81ac}
 input{font:inherit;font-size:13px;background:#1b1f26;color:#d8dee9;border:1px solid #3b4252;
       border-radius:4px;padding:6px 9px;width:340px;max-width:55vw}
 .bar{margin-bottom:16px} #msg{margin-left:10px;font-size:12px}
 .tab{background:#22262d;border-color:#22262d} .tab.on{background:#5e81ac;border-color:#5e81ac}
 .warn{color:#ebcb8b}
</style></head><body>
<h1>library <a href="/" style="font-size:12px">&larr; jobs</a></h1>
<div class="sub" id="counts">reading the library&hellip;</div>
<div class="bar">
  <input id="q" placeholder="filter by title or filename" oninput="render()">
  <button class="tab on" id="t-all" onclick="setFilter('all')">all</button>
  <button class="tab" id="t-missing" onclick="setFilter('missing')">missing</button>
  <button class="tab" id="t-translated" onclick="setFilter('translated')">translated</button>
  <button onclick="load(true)">rescan</button>
  <span id="msg"></span>
</div>
<table><thead><tr><th>show / film</th><th>file</th><th>state</th><th></th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const hdrs = TOKEN ? {"x-api-token": TOKEN, "Content-Type": "application/json"}
                   : {"Content-Type": "application/json"};
let ITEMS = [], FILTER = "all", SOURCE = "";
function say(t, ok) {
  const m = document.getElementById("msg");
  m.textContent = t; m.style.color = ok ? "#a3be8c" : "#bf616a";
}
function esc(s) { return String(s).replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]); }
function setFilter(f) {
  FILTER = f;
  for (const t of ["all", "missing", "translated"])
    document.getElementById("t-" + t).classList.toggle("on", t === f);
  render();
}
async function load(refresh) {
  say(refresh ? "rescanning the library..." : "loading...", true);
  const r = await fetch("/api/library" + (refresh ? "?refresh=true" : ""), {headers: hdrs});
  if (!r.ok) return say("not authorised - open this page with ?token=...", false);
  const d = await r.json();
  ITEMS = d.items; SOURCE = d.source;
  say("", true); render();
}
function render() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  let rows = ITEMS.filter(i => !q || i.title.toLowerCase().includes(q)
                                  || i.name.toLowerCase().includes(q));
  if (FILTER === "missing") rows = rows.filter(i => !i.translated);
  if (FILTER === "translated") rows = rows.filter(i => i.translated);

  const done = ITEMS.filter(i => i.translated).length;
  document.getElementById("counts").innerHTML =
    `${ITEMS.length} videos &middot; ${done} translated &middot; ` +
    `${ITEMS.length - done} without a subtitle &middot; via ${esc(SOURCE)}`;

  document.getElementById("rows").innerHTML = rows.slice(0, 1500).map(i => {
    const state = i.translated ? `<span class="done">${esc(i.subtitle)}</span>`
                : i.pending    ? '<span class="pending">queued</span>'
                :                '<span class="missing">none</span>';
    // Already translated is not a reason to refuse - it is a reason to warn.
    // force renames the existing file to .bak rather than deleting it.
    const dis = i.pending ? "disabled" : "";
    const redo = i.translated ? " (redo)" : "";
    return `<tr><td>${esc(i.title)}</td>
      <td class="path">${esc(i.name)}</td><td>${state}</td>
      <td><button ${dis} title="run now on the local model, ahead of the queue"
        onclick="go(this,'${encodeURIComponent(i.video)}','ollama',${!!i.translated})">local now${redo}</button>
      <button class="claude" ${dis} title="run now on Claude, ahead of the queue"
        onclick="go(this,'${encodeURIComponent(i.video)}','anthropic',${!!i.translated})">Claude now${redo}</button></td></tr>`;
  }).join("") || '<tr><td colspan="4">nothing matches</td></tr>';
}
async function go(btn, video, provider, translated) {
  const path = decodeURIComponent(video);
  if (translated && !confirm(
      "This already has a subtitle.\\n\\nRe-translate it? The existing file is " +
      "renamed to .bak, not deleted.")) return;
  btn.disabled = true; say("queueing...", true);
  // Clicking a specific item always means "now" - the queue exists for the
  // sweeper and the bazarr webhook, not for something you picked by hand.
  const body = {video: path, force: !!translated, provider: provider, rush: true};
  const r = await fetch("/translate", {method: "POST", headers: hdrs,
                                        body: JSON.stringify(body)});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) { btn.disabled = false; return say(d.detail || r.status, false); }
  say(d.queued ? `queued as job ${d.job}` : d.reason, d.queued);
}
load(false);
</script></body></html>"""


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request):
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)
    return LIBRARY_PAGE



BACKENDS_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>tarjem backends</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#14161a;
      color:#d8dee9;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} a{color:#88c0d0}
 .sub{color:#7b8494;margin-bottom:20px}
 table{border-collapse:collapse;width:100%;max-width:1000px;font-size:13px}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #262a31}
 th{color:#7b8494;font-weight:500}
 .on{color:#a3be8c} .off{color:#7b8494} .bad{color:#bf616a} .busy{color:#ebcb8b}
 button{font:inherit;font-size:12px;background:#3b4252;color:#e5e9f0;border:1px solid #4c566a;
        border-radius:4px;padding:4px 10px;cursor:pointer;margin-right:4px}
 button:hover{background:#4c566a} button:disabled{opacity:.45;cursor:default}
 button.danger{border-color:#bf616a}
 input,select{font:inherit;font-size:13px;background:#1b1f26;color:#d8dee9;
   border:1px solid #3b4252;border-radius:4px;padding:6px 8px;margin-right:6px}
 input.url{width:280px} input.model{width:210px}
 .add{margin-top:26px;padding:14px;background:#1b1f26;border:1px solid #262a31;border-radius:6px;
      max-width:1000px}
 .hint{color:#7b8494;font-size:12px;margin-top:8px}
 #msg{margin-left:10px;font-size:12px}
</style></head><body>
<h1>backends <a href="/" style="font-size:12px">&larr; jobs</a>
  <a href="/library" style="font-size:12px">library</a></h1>
<div class="sub">Machines that do the translating. Turn one off to get its GPU
  back - a job already running on it finishes first.</div>

<table><thead><tr><th>backend</th><th>model</th><th>state</th><th></th></tr></thead>
<tbody id="rows"><tr><td colspan="4">loading&hellip;</td></tr></tbody></table>

<div class="add">
  <select id="kind">
    <option value="ollama">ollama</option>
    <option value="openai">openai-compatible (LM Studio, vLLM, ...)</option>
    <option value="anthropic">anthropic</option>
  </select>
  <input class="url" id="url" placeholder="http://192.168.1.50:11434" spellcheck="false">
  <input class="model" id="model" placeholder="command-r7b-arabic" spellcheck="false">
  <button onclick="add()">add backend</button>
  <span id="msg"></span>
  <div class="hint">An ollama backend uses the native API, which enforces the
    output schema. Anything else goes through the OpenAI-compatible path.</div>
</div>

<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const hdrs = TOKEN ? {"x-api-token": TOKEN, "Content-Type": "application/json"}
                   : {"Content-Type": "application/json"};
function say(t, ok) {
  const m = document.getElementById("msg");
  m.textContent = t; m.style.color = ok ? "#a3be8c" : "#bf616a";
}
function esc(s) { return String(s).replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]); }
async function api(method, path, body) {
  const r = await fetch(path, {method, headers: hdrs,
                               body: body ? JSON.stringify(body) : null});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.status);
  return d;
}
function render(list) {
  document.getElementById("rows").innerHTML = list.map(b => {
    let state = b.enabled ? '<span class="on">enabled</span>'
                          : '<span class="off">disabled</span>';
    if (b.enabled && !b.healthy)
      state += ` <span class="bad">unreachable, retrying in ${b.down_for_s}s</span>`;
    if (b.busy) state += ' <span class="busy">&middot; translating</span>';
    return `<tr>
      <td>${esc(b.name)}<br><span class="off">${esc(b.kind)}</span></td>
      <td>${esc(b.model || "(default)")}</td>
      <td>${state}</td>
      <td>
        <button onclick="toggle('${esc(b.name)}',${!b.enabled})">
          ${b.enabled ? "disable" : "enable"}</button>
        <button class="danger" ${b.busy ? "disabled" : ""}
          onclick="remove('${esc(b.name)}')">remove</button>
      </td></tr>`;
  }).join("") || '<tr><td colspan="4">no backends - tarjem will use LLM_PROVIDER instead</td></tr>';
}
async function load() {
  try { render((await api("GET", "/api/backends")).backends); }
  catch (e) { say("not authorised - open this page with ?token=...", false); }
}
async function toggle(name, enabled) {
  try { render((await api("PATCH", `/api/backends/${encodeURIComponent(name)}?enabled=${enabled}`)).backends);
        say(`${name} ${enabled ? "enabled" : "disabled"}`, true); }
  catch (e) { say(String(e.message), false); }
}
async function remove(name) {
  if (!confirm(`Remove ${name}? Its settings are lost; disable instead to keep them.`)) return;
  try { render((await api("DELETE", `/api/backends/${encodeURIComponent(name)}`)).backends);
        say(`${name} removed`, true); }
  catch (e) { say(String(e.message), false); }
}
async function add() {
  const url = document.getElementById("url").value.trim();
  if (!url) return say("a url is required", false);
  try {
    const d = await api("POST", "/api/backends", {
      kind: document.getElementById("kind").value,
      url, model: document.getElementById("model").value.trim()});
    render(d.backends); say(`added ${d.added}`, true);
    document.getElementById("url").value = "";
  } catch (e) { say(String(e.message), false); }
}
load();
setInterval(load, 10000);
</script></body></html>"""


@app.get("/backends", response_class=HTMLResponse)
def backends_page(request: Request):
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)
    return BACKENDS_PAGE


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, status: str = ""):
    # A browser gets the sign-in page rather than a bare 401.
    if not auth_mod.authenticated(settings, request):
        return RedirectResponse("/login", status_code=303)

    db = store()
    counts = db.counts()
    # Each count is a filter link - with a long queue the default view hides
    # those rows, so there has to be a way back to them.
    pills = "".join(
        f'<a class="pill {k}" href="/?status={k}">{k}: {v}</a>'
        for k, v in sorted(counts.items())
    ) or '<span class="pill">no jobs yet</span>'

    rows = []
    # Default view hides the queue: with a real backlog it is dozens of
    # identical rows, and it buries both the running job and the history.
    rows_source = (db.recent(60, status=status, running_first=True) if status
                   else db.recent(40, running_first=True, exclude=("queued",)))
    for j in rows_source:
        pct = int((j.get("progress") or 0) * 100)
        stats = j.get("stats") or {}
        if j["status"] == "done" and stats:
            result = (f'{stats.get("translated", 0)}/{stats.get("total_cues", 0)} cues '
                      f'&middot; {stats.get("seconds", 0)}s')
        elif j["status"] == "failed":
            result = f'<span class="err">{_esc(j.get("error", ""))}</span>'
        else:
            result = _esc(j.get("stage", ""))
        # Rushing something already queued or running would only duplicate it.
        action = ""
        if j["status"] in ("failed", "done", "skipped"):
            action = (
                f'<button onclick="rushJob(this,{j["id"]},&quot;ollama&quot;)" '
                f'title="re-run on the local model, ahead of the queue">local</button>'
                f'<button onclick="rushJob(this,{j["id"]},&quot;anthropic&quot;)" '
                f'title="re-run on Claude, ahead of the queue">Claude</button>')
        badge = ' <span class="pill">rush</span>' if j.get("priority") else ""

        rows.append(
            f'<tr><td>{j["id"]}</td>'
            f'<td>{_esc(j.get("title") or Path(j["video"]).name)}{badge}<br>'
            f'<span class="path">{_esc(Path(j["video"]).name)}</span></td>'
            f'<td class="{j["status"]}">{j["status"]}</td>'
            f'<td><span class="bar"><i style="width:{pct}%"></i></span> {pct}%</td>'
            f'<td>{_esc(j.get("provider") or j.get("origin") or "")}</td>'
            f'<td>{result}</td><td>{action}</td></tr>'
        )

    keyhint = ("" if settings.anthropic_api_key else
               " <b>ANTHROPIC_API_KEY is not set</b>, so this will fail until it is.")
    if settings.api_token:
        keyhint += " Open this page with ?token=... for the buttons to authenticate."

    return PAGE.format(
        provider=settings.provider,
        model=settings.active_model,
        target=settings.target_lang,
        register=settings.register,
        bazarr="ok" if state["bazarr"].ping() else "unreachable",
        pills=pills + (
            ' <a class="pill" href="/">show all activity</a>' if status
            else ' <span class="pill" style="background:none;color:#7b8494">'
                 'queued rows hidden - click a count to filter</span>'),
        keyhint=keyhint,
        logout=('<form method="post" action="/logout" style="display:inline">'
                '<button style="font-size:11px">sign out</button></form>'
                if settings.auth_enabled else ''),
        rows="\n".join(rows) or '<tr><td colspan="7">nothing yet</td></tr>',
    )


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
