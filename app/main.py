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
    for worker in workers:
        worker.start()

    sweeper = Sweeper(settings, store, bazarr)
    if settings.sweep_enabled:
        sweeper.start()

    state.update(store=store, bazarr=bazarr, workers=workers, sweeper=sweeper, started=time.time())
    log.info("tarjem up | provider=%s model=%s target=%s register=%s",
             settings.provider, settings.model, settings.target_lang, settings.register)
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
        "model": settings.model if settings.provider == "anthropic" else settings.openai_model,
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

    job_id = store().enqueue(
        str(video), req.title or display_title(video), series_key(video), trigger="manual"
    )
    if job_id is None:
        return JSONResponse({"queued": False, "reason": "already queued"})
    if req.subtitle:
        store().update(job_id, detail=req.subtitle)
    return JSONResponse({"queued": True, "job": job_id})


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
    if not store().requeue(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    return {"requeued": True}


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
</style></head><body>
<h1>tarjem</h1>
<div class="sub">{provider} &middot; {model} &middot; {target}/{register} &middot; bazarr {bazarr}</div>
<div style="margin-bottom:16px">{pills}</div>
<table><tr><th>#</th><th>title</th><th>status</th><th>progress</th><th>source</th><th>result</th></tr>
{rows}
</table></body></html>"""


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
        rows.append(
            f'<tr><td>{j["id"]}</td>'
            f'<td>{_esc(j.get("title") or Path(j["video"]).name)}<br>'
            f'<span class="path">{_esc(Path(j["video"]).name)}</span></td>'
            f'<td class="{j["status"]}">{j["status"]}</td>'
            f'<td><span class="bar"><i style="width:{pct}%"></i></span> {pct}%</td>'
            f'<td>{_esc(j.get("origin") or "")}</td>'
            f'<td>{result}</td></tr>'
        )

    return PAGE.format(
        provider=settings.provider,
        model=settings.model if settings.provider == "anthropic" else settings.openai_model,
        target=settings.target_lang,
        register=settings.register,
        bazarr="ok" if state["bazarr"].ping() else "unreachable",
        pills=pills,
        rows="\n".join(rows) or '<tr><td colspan="6">nothing yet</td></tr>',
    )


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
