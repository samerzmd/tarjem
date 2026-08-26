"""The shared page shell.

Built to sit alongside Sonarr, Radarr and Bazarr without looking like a
different application: the same dark palette, the same fixed left rail, the
same blue accent. Every page is this shell plus a body, so the chrome is
written once.

Templates here are concatenated rather than str.format'ed - CSS and JS are full
of braces, and escaping every one of them for the sake of two substitutions
makes the source unreadable.
"""
from __future__ import annotations

CSS = """
:root{
  --bg:#1c1c1c; --panel:#2a2a2a; --panel-2:#333; --border:#3a3a3a;
  --text:#ccc; --muted:#8c8c8c; --accent:#5799ef; --accent-dim:#3a6ea8;
  --green:#27c24c; --red:#f05050; --orange:#ffa500; --purple:#9b59b6;
  --rail:210px;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--text);
  font:14px/1.5 "Roboto","Helvetica Neue",Helvetica,Arial,sans-serif;
}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}

/* ---- left rail ---- */
.rail{
  position:fixed; left:0; top:0; bottom:0; width:var(--rail);
  background:#262626; border-right:1px solid var(--border); overflow-y:auto;
}
.brand{
  display:flex; align-items:center; gap:9px; height:60px; padding:0 16px;
  border-bottom:1px solid var(--border); color:#fff; font-size:17px;
  font-weight:500; letter-spacing:.4px;
}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);flex:none}
.nav{padding:10px 0}
.nav a{
  display:flex; align-items:center; gap:11px; padding:9px 16px;
  color:var(--text); font-size:14px; border-left:3px solid transparent;
}
.nav a:hover{background:#2e2e2e;text-decoration:none}
.nav a.on{background:#2f2f2f;border-left-color:var(--accent);color:#fff}
.nav svg{width:16px;height:16px;flex:none;opacity:.85}
.railfoot{padding:14px 16px;color:var(--muted);font-size:11px;line-height:1.7}

/* ---- top bar ---- */
.top{
  position:sticky; top:0; z-index:5; height:60px; display:flex; align-items:center;
  gap:14px; padding:0 22px; background:var(--panel);
  border-bottom:1px solid var(--border);
}
.top h1{font-size:17px;font-weight:500;margin:0;color:#fff}
.top .spacer{flex:1}
.wrap{margin-left:var(--rail)}
.content{padding:22px}

/* ---- pills ---- */
.pill{
  display:inline-block; padding:2px 9px; border-radius:11px; font-size:12px;
  background:var(--panel-2); color:var(--text); margin-right:6px;
}
a.pill:hover{background:#3f3f3f;text-decoration:none}
.pill.done,.pill.enabled{background:rgba(39,194,76,.16);color:var(--green)}
.pill.failed{background:rgba(240,80,80,.16);color:var(--red)}
.pill.running{background:rgba(255,165,0,.16);color:var(--orange)}
.pill.queued{background:rgba(87,153,239,.16);color:var(--accent)}
.pill.skipped{background:#333;color:var(--muted)}

/* ---- panels and tables ---- */
.panel{
  background:var(--panel); border:1px solid var(--border); border-radius:4px;
  margin-bottom:18px;
}
.panel h2{
  font-size:13px; font-weight:500; text-transform:uppercase; letter-spacing:.6px;
  color:var(--muted); margin:0; padding:12px 16px; border-bottom:1px solid var(--border);
}
.panel .inner{padding:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
thead th{
  text-align:left; padding:10px 14px; color:var(--muted); font-weight:500;
  font-size:12px; text-transform:uppercase; letter-spacing:.5px;
  background:#2f2f2f; border-bottom:1px solid var(--border);
  position:sticky; top:60px;
}
tbody td{padding:10px 14px;border-bottom:1px solid #333;vertical-align:middle}
tbody tr:hover{background:#2f2f2f}
td.sub,.sub{color:var(--muted);font-size:12px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}

/* ---- controls ---- */
button{
  font:inherit; font-size:13px; background:var(--panel-2); color:var(--text);
  border:1px solid #454545; border-radius:3px; padding:5px 12px; cursor:pointer;
  margin-right:5px;
}
button:hover{background:#404040}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{background:#6ba7f2}
button.danger{border-color:#7a3a3a;color:#e08585}
button.danger:hover{background:#4a2a2a}
button.tab{background:transparent;border-color:transparent;color:var(--muted)}
button.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
input,select{
  font:inherit; font-size:13px; background:#1f1f1f; color:var(--text);
  border:1px solid #454545; border-radius:3px; padding:6px 10px; margin-right:6px;
}
input:focus,select:focus{outline:none;border-color:var(--accent)}
input::placeholder{color:#6b6b6b}

/* ---- misc ---- */
.bar{display:inline-block;height:5px;width:110px;background:#1f1f1f;border-radius:3px;
     overflow:hidden;vertical-align:middle;margin-right:7px}
.bar i{display:block;height:100%;background:var(--accent)}
.msg{margin-left:10px;font-size:12px}
.ok{color:var(--green)} .err{color:var(--red)} .warn{color:var(--orange)}
.empty{padding:26px;text-align:center;color:var(--muted)}
.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:16px}
@media(max-width:760px){
  :root{--rail:0px}
  .rail{display:none}
  thead th{top:0}
}
"""

ICONS = {
    "jobs": '<path d="M3 4h18v2H3zM3 9h18v2H3zM3 14h12v2H3zM3 19h12v2H3z"/>',
    "library": '<path d="M4 3h4v18H4zM10 3h4v18h-4zM17 3.5l3.8 17.1-2 .4L15 3.9z"/>',
    "backends": '<path d="M3 4h18v6H3zM3 14h18v6H3zM6 6.5h2v1H6zM6 16.5h2v1H6z"/>',
}


def _nav(active: str) -> str:
    items = (("jobs", "/", "Activity"),
             ("library", "/library", "Library"),
             ("backends", "/backends", "Backends"))
    out = []
    for key, href, label in items:
        cls = " class='on'" if key == active else ""
        out.append(
            f"<a href='{href}'{cls}>"
            f"<svg viewBox='0 0 24 24' fill='currentColor'>{ICONS[key]}</svg>"
            f"{label}</a>"
        )
    return "".join(out)


def shell(*, title: str, active: str, heading: str, body: str,
          script: str = "", footer: str = "", refresh: int = 0) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"{meta}<style>{CSS}</style></head><body>"
        "<div class='rail'>"
        "<div class='brand'><span class='dot'></span>tarjem</div>"
        f"<nav class='nav'>{_nav(active)}</nav>"
        f"<div class='railfoot'>{footer}</div>"
        "</div>"
        "<div class='wrap'>"
        f"<div class='top'><h1>{heading}</h1><div class='spacer'></div></div>"
        f"<div class='content'>{body}</div>"
        "</div>"
        f"<script>{script}</script></body></html>"
    )


LOGIN = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>tarjem</title><meta name='viewport' content='width=device-width,initial-scale=1'>
<style>__CSS__
 body{display:flex;align-items:center;justify-content:center;height:100vh}
 form{background:var(--panel);border:1px solid var(--border);border-radius:4px;
      padding:30px;width:310px}
 .brand{height:auto;border:0;padding:0 0 4px;font-size:20px}
 form p{color:var(--muted);font-size:12px;margin:0 0 20px}
 form input{width:100%;margin:0}
 form button{width:100%;margin:14px 0 0;padding:9px}
</style></head><body>
<form method='post' action='/login'>
  <div class='brand'><span class='dot'></span>tarjem</div>
  <p>AI Arabic subtitles</p>
  <input type='password' name='password' placeholder='Password' autofocus
         autocomplete='current-password'>
  <button class='primary' type='submit'>Sign in</button>
  __ERROR__
</form></body></html>""".replace("__CSS__", CSS)


# Shared client-side helpers. Every page talks to the same API and needs the
# same three things: forward the token, escape text, and report what happened.
JS_BASE = """
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const HDRS = TOKEN ? {"x-api-token": TOKEN, "Content-Type": "application/json"}
                   : {"Content-Type": "application/json"};
function say(t, ok) {
  const m = document.getElementById("msg");
  if (!m) return;
  m.textContent = t;
  m.className = "msg " + (ok ? "ok" : "err");
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]);
}
async function api(method, path, body) {
  const r = await fetch(path, {method, headers: HDRS,
                               body: body ? JSON.stringify(body) : null});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
  return d;
}
"""
