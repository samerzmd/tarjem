"""HTTP surface: the Bazarr webhook, a manual trigger, and a status page."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import sources
from .bazarr import BazarrClient
from .config import settings
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
    workers = [Worker(settings, store, bazarr, name=f"worker-{i + 1}")
               for i in range(max(1, settings.workers))]
    # A lane of its own for rush jobs. They run on whichever provider the
    # request named - usually a cloud one - so they don't contend for the same
    # hardware as the local model, and shouldn't wait behind it either.
    workers.append(Worker(settings, store, bazarr, name="rush", priority_only=True))
    for worker in workers:
        worker.start()

    sweeper = Sweeper(settings, store, bazarr)
    if settings.sweep_enabled:
        sweeper.start()

    state.update(store=store, bazarr=bazarr, workers=workers, sweeper=sweeper, started=time.time())
    log.info("tarjem up | provider=%s model=%s target=%s register=%s",
             settings.provider, settings.active_model, settings.target_lang, settings.register)
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


def auth(x_api_token: str = Header(default=""), token: str = Query(default="")) -> None:
    if settings.api_token and settings.api_token not in (x_api_token, token):
        raise HTTPException(status_code=401, detail="bad or missing token")


def store() -> Store:
    return state["store"]


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
def health() -> dict:
    db = store()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - state["started"]),
        "provider": settings.provider,
        "model": settings.active_model,
        "target_lang": settings.target_lang,
        "register": settings.register,
        "bazarr": state["bazarr"].ping(),
        "jobs": db.counts(),
        "dry_run": settings.dry_run,
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
    return state["sweeper"].sweep(limit or None)


@app.get("/jobs", dependencies=[Depends(auth)])
def jobs(limit: int = Query(default=50, ge=1, le=500), status: str | None = None) -> dict:
    return {"jobs": store().recent(limit, status), "counts": store().counts()}


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
 .pill{{background:#22262d;padding:2px 8px;border-radius:10px;margin-right:6px}}
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
<h1>tarjem</h1>
<div class="sub">{provider} &middot; {model} &middot; {target}/{register} &middot; bazarr {bazarr}</div>
<div style="margin-bottom:16px">{pills}</div>

<div class="rushbar">
  <input id="path" placeholder="/media/tv/Show/Season 1/Episode.mkv"
         value="" spellcheck="false">
  <button onclick="rushPath()">Translate now on Claude</button>
  <span id="msg"></span>
  <div class="hint">Jumps the queue and runs on Claude in its own lane, so it does
    not wait behind the local model.{keyhint}</div>
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
async function rushPath() {{
  const v = document.getElementById("path").value.trim();
  if (!v) return say("paste a path first", false);
  say("queueing...", true);
  try {{
    const d = await post("/translate", {{video: v, provider: "anthropic", rush: true, force: true}});
    say(d.queued ? `queued as job ${{d.job}}` : d.reason, d.queued);
  }} catch (e) {{ say(String(e.message), false); }}
}}
async function rushJob(btn, id) {{
  btn.disabled = true;
  say("queueing...", true);
  try {{
    const d = await post(`/jobs/${{id}}/rush?provider=anthropic`);
    say(`queued as job ${{d.job}} on Claude`, true);
  }} catch (e) {{ btn.disabled = false; say(String(e.message), false); }}
}}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    db = store()
    counts = db.counts()
    pills = "".join(
        f'<span class="pill {k}">{k}: {v}</span>' for k, v in sorted(counts.items())
    ) or '<span class="pill">no jobs yet</span>'

    rows = []
    for j in db.recent(40):
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
            action = (f'<button onclick="rushJob(this,{j["id"]})" '
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
        pills=pills,
        keyhint=keyhint,
        rows="\n".join(rows) or '<tr><td colspan="7">nothing yet</td></tr>',
    )


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
