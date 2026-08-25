# tarjem

AI Arabic subtitles for the *arr stack. When Bazarr can't find Arabic, tarjem takes
whatever subtitle *does* exist — a sidecar, a track buried in the MKV, or a Whisper
transcript — and translates it, writing `Movie.ar.srt` next to the video where
Jellyfin and Bazarr both pick it up.

It automates the loop you're already doing by hand, and does three things that
pasting an SRT into a chat window can't:

- **A glossary pass first.** Before translating a line, the model reads a sample
  drawn from across the whole file and fixes the Arabic rendering of every
  recurring name, place and coined term. That brief is cached and reused for
  every later episode of the same series, so a character's name doesn't change
  spelling in episode 4.
- **A rolling context window.** Each batch sees the previous cues *and their
  accepted translations*, so pronouns, honorifics and running jokes carry across
  batch boundaries instead of resetting every few hundred lines.
- **Structured id round-tripping.** The model returns cue ids, not a blob. A
  missing or extra line is detected, repaired, and — if repair fails — narrowed
  to that one cue. Timings are never touched, so a bad batch can't desync a file.

---

## How it plugs into your stack

```
Sonarr/Radarr ──> Bazarr ──(finds English, fires post-processing)──> tarjem
                    ^                                                  │
                    └──────────(rescan: the .ar.srt is there)──────────┘
                                                                       │
                                       /mnt/storage/media/.../X.ar.srt ┘
                                                    │
                                                 Jellyfin
```

Two triggers, and you want both:

| Trigger | Fires when | Covers |
|---|---|---|
| **Webhook** | Bazarr downloads any non-Arabic subtitle | New downloads, automatically |
| **Sweeper** | Every `SWEEP_INTERVAL_MIN` | Your existing library, and anything the webhook missed |

The sweeper asks Bazarr's own `wanted` API what's still missing Arabic, so it
respects your language profiles and exclusions. Set `SWEEP_SOURCE=disk` to walk
`MEDIA_ROOTS` looking for videos with no `.ar.srt` instead.

---

## Setup

### 1. Configure

```bash
cp .env.example .env
```

Fill in `ANTHROPIC_API_KEY` and `BAZARR_API_KEY` (Bazarr → Settings → General →
Security → API key). Set `TARJEM_TOKEN` to any random string if you want the
webhook authenticated.

### 2. Add it to the stack

The compose file here joins the existing `media-server` network and mounts media
at exactly the same paths Bazarr uses, so paths in the webhook resolve without
translation:

```bash
docker compose up -d --build
```

Or paste the service block into `media-server/docker-compose.yml` and bring the
whole stack up together — see [`docker-compose.yml`](docker-compose.yml) for the
block. Either way, `user: "1000:1000"` must match the `PUID`/`PGID` the rest of
your stack runs as, or the sidecars land with the wrong owner.

Check it came up: `http://YOUR-SERVER:8081/` shows a live job list.

### 3. Make Bazarr download English

This is the step that's easy to miss, and it's the one that decides whether any
of this works. **If your language profile only wants Arabic, Bazarr never
downloads anything for tarjem to translate** — a failed Arabic search leaves you
with no file at all, not an English one.

Bazarr → Settings → Languages → the profile your library uses → the pencil icon.
Under **Languages**, click **Add Language** and add English alongside Arabic:

| ID | Language | Subtitles Type | Search only when... |
|---|---|---|---|
| 1 | Arabic | Normal or hearing-impaired | Always |
| 2 | English | Normal or hearing-impaired | **Always** |

Three settings in that dialog will silently defeat the whole thing if they're wrong:

- **Cutoff must be empty — not `Any`.** Cutoff means "stop searching once you
  have this". `Any` counts *every* language as satisfying it, so Bazarr gives up
  on Arabic the moment English lands. The field is clearable; clear it.
- **"Search only when..." must be `Always` on the English row.** Set to `no audio
  track matches`, Bazarr skips English subtitles for anything with English audio
  — which is most live-action libraries.
- **"Use Original Format" (under *Subtitles*) should stay off**, so you get clean
  `.srt` sidecars. tarjem converts `.ass`/`.ssa` with ffmpeg anyway, so this is a
  preference rather than a requirement.

Leave **Subtitles Type** as *Normal or hearing-impaired*: an SDH track is more
source text to translate from, and `STRIP_HI=true` removes the `[door creaks]`
annotations on the way out.

Expect a burst of English searches across the library right after you save — the
profile applies to every series and film already assigned to it. Run tarjem on a
single file first (below) and check you like the Arabic before opening that tap.

### 4. Wire the post-processing hook

Bazarr → Settings → Subtitles → **Custom Post-Processing** → enable, and paste:

```
curl -sS -m 30 -X POST http://tarjem:8080/hook/bazarr -H "x-api-token: YOUR_TARJEM_TOKEN" --data-urlencode video={{episode}} --data-urlencode subtitle={{subtitles}} --data-urlencode lang={{subtitles_language_code2}} --data-urlencode series_id={{series_id}} --data-urlencode episode_id={{episode_id}}
```

Drop the `-H "x-api-token: ..."` part if you left `TARJEM_TOKEN` empty.

**Do not put quotes around the `{{...}}` placeholders.** Bazarr already wraps
each value in double quotes when it substitutes, and it strips one quote
character on either side of the placeholder if you add your own.

**Do not use a JSON body here.** Bazarr runs this through `sh -c`, and a title
with an apostrophe in it — `Ocean's Eleven` — breaks single-quoted JSON at the
shell before curl ever sees it. The `--data-urlencode` form above survives it;
tarjem accepts both form-encoded and JSON posts, but only this one is safe.

tarjem ignores hooks where the downloaded language *is* Arabic, so it won't chase
its own output.

### 5. Backfill the existing library

```bash
curl -X POST "http://YOUR-SERVER:8081/sweep?limit=5" -H "x-api-token: YOUR_TOKEN"
```

Start with a small limit and look at the results before opening the tap.
`SWEEP_LIMIT` caps how many one automatic sweep will queue, so a first run can't
work through your whole library in an afternoon.

---

## Try it on one file first

Before wiring anything up, run a single file through and read the output:

```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... python -m app.cli "Movie.en.srt" --limit 60
```

`--limit 60` translates just the first 60 cues so you can judge the register
cheaply. `--register gulf` (or `egyptian`, `levantine`, `msa-light`) changes the
dialect. It writes `Movie.ar.srt` next to the input and prints token usage.

To pull the source straight out of an MKV:

```bash
python -m app.cli "Movie.mkv" --from-video --limit 60
```

---

## Where the source subtitle comes from

In order, first hit wins:

1. **A sidecar** next to the video in one of `SOURCE_LANGS` — `Movie.en.srt`,
   `Movie.eng.srt`, `.ass`/`.ssa`/`.vtt` converted on the fly. Forced tracks are
   ranked last (they're signage only, a few dozen cues); anything already Arabic
   is skipped.
2. **An embedded text track**, extracted with ffmpeg. Bitmap tracks (PGS, VOBSUB)
   are skipped — those need OCR, which is out of scope.
3. **Whisper**, if you set `WHISPER_URL` to a whisper-asr-webservice or subgen
   endpoint. Only used when there's no text subtitle at all.

---

## Cost

Each job reports its real token usage — `GET /jobs/{id}` shows `usage`, including
how much was served from cache. Use that rather than an estimate.

As a rough order of magnitude on a ~1,200-cue film: low single-digit dollars on
`claude-opus-5`, noticeably less on `claude-sonnet-5`, and a 22-minute episode
runs perhaps a third of a film. Two settings move this the most:

- `LLM_MODEL=claude-sonnet-5` — meaningfully cheaper per file. Opus is the better
  translator, especially for idiom and register; Sonnet is a reasonable trade for
  a large backfill.
- `LLM_EFFORT=low` (the default) — translation isn't a reasoning task, and higher
  effort mostly buys thinking tokens you're paying for and not using.

The glossary and the translation rules sit in a cached prompt prefix, so most of
the per-batch input on a given file is billed at the cache-read rate.

---

## API

Everything takes `x-api-token` as a header, or `?token=` in the query string.

| Method | Path | |
|---|---|---|
| `GET` | `/` | Status page, auto-refreshing |
| `GET` | `/health` | Provider, Bazarr reachability, job counts |
| `POST` | `/hook/bazarr` | The Bazarr post-processing webhook |
| `POST` | `/translate` | `{"video": "...", "force": false}` — queue one file |
| `POST` | `/sweep?limit=N` | Run a sweep now |
| `GET` | `/jobs?limit=50&status=failed` | Job list |
| `GET` | `/jobs/{id}` | One job, with stats and token usage |
| `POST` | `/jobs/{id}/retry` | Requeue |
| `GET` | `/jobs/{id}/subtitle` | The produced SRT, as text |
| `GET` | `/glossaries` | Cached per-title briefs |
| `GET`/`DELETE` | `/glossaries/{key}` | Inspect or drop one |

Re-translate a file you didn't like: `POST /translate` with `{"force": true}` —
the existing `.ar.srt` is renamed to `.bak` rather than deleted. Drop that title's
glossary first if the problem was a character's name.

---

## Tuning the output

| Setting | Effect |
|---|---|
| `ARABIC_REGISTER` | `msa` (default), `msa-light`, `gulf`, `egyptian`, `levantine` |
| `MAX_LINE_CHARS` / `MAX_LINES` | Caption width discipline. 42×2 is the broadcast norm |
| `BATCH_SIZE` | Cues per request. Lower = more context per cue, more calls |
| `CONTEXT_CUES` | How many previous cues+translations each batch sees |
| `GLOSSARY_ENABLED` | Turn off to skip the brief pass |
| `STRIP_HI` | Drop `[door creaks]` / `SPEAKER:` when the only source is an SDH track |
| `DRY_RUN` | Translate and report, write nothing |

---

## Troubleshooting

**Hook fires but nothing queues.** Look at `docker logs bazarr` for the curl
output. A `404 video not found` means Bazarr and tarjem disagree about paths —
set `PATH_MAP=/bazarr/path:/tarjem/path`. `401` means the token doesn't match.

**"no usable source subtitle found".** Nothing was on disk and nothing text-based
was in the container. Check with
`docker exec tarjem ffprobe -select_streams s -show_streams "/media/movies/.../X.mkv"`.
If the only tracks are `hdmv_pgs_subtitle`, that's a bitmap track — set
`WHISPER_URL` and let Whisper transcribe instead.

**Arabic never gets requested.** Confirm the language profile actually lists
Arabic *and* English, and that the profile is assigned to the series or film.

**Names drift between episodes.** Check `GET /glossaries` — if the series has no
entry, the brief pass failed. If it has a wrong entry, `DELETE` it and re-run.

**Nothing appears in Bazarr.** tarjem asks Bazarr to rescan only when the webhook
or sweep gave it an item id. Otherwise Bazarr picks the file up on its next scan;
the file is already on disk either way.

---

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests -q
```

The tests stub the provider — no API key, no network. They cover SRT round-trip
fidelity, batch repair when the model drops a cue, markup preservation, and the
webhook shapes Bazarr actually sends.
