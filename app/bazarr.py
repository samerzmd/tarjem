"""Bazarr API client.

Used for two things: asking Bazarr what is still missing Arabic (it already knows,
and it respects your language profiles and exclusions), and telling it to rescan
the folder once a translation lands so the new sidecar shows up in the UI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class WantedItem:
    kind: str            # "movie" | "episode"
    item_id: int         # radarrId or sonarrEpisodeId
    series_id: int       # sonarrSeriesId, 0 for movies
    title: str
    path: str = ""


class BazarrClient:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.enabled = bool(cfg.bazarr_url and cfg.bazarr_api_key)
        self._client = httpx.Client(
            base_url=f"{cfg.bazarr_url}/api",
            headers={"X-API-KEY": cfg.bazarr_api_key},
            timeout=60.0,
        )

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            return self._client.get("/system/status").status_code == 200
        except httpx.RequestError:
            return False

    # -- discovery --------------------------------------------------------

    def wanted(self, lang: str) -> list[WantedItem]:
        """Every movie and episode Bazarr still wants ``lang`` for."""
        if not self.enabled:
            return []
        return self._wanted_movies(lang) + self._wanted_episodes(lang)

    def _wanted_movies(self, lang: str) -> list[WantedItem]:
        data = self._get("/movies/wanted", {"start": 0, "length": -1})
        items: list[WantedItem] = []
        for row in data.get("data", []):
            if not _wants(row, lang):
                continue
            items.append(WantedItem("movie", row.get("radarrId", 0), 0, row.get("title", "")))
        self._fill_paths(items, "movie")
        return items

    def _wanted_episodes(self, lang: str) -> list[WantedItem]:
        data = self._get("/episodes/wanted", {"start": 0, "length": -1})
        items: list[WantedItem] = []
        for row in data.get("data", []):
            if not _wants(row, lang):
                continue
            name = " - ".join(
                p for p in (row.get("seriesTitle"), row.get("episode_number"), row.get("episodeTitle")) if p
            )
            items.append(WantedItem(
                "episode",
                row.get("sonarrEpisodeId", 0),
                row.get("sonarrSeriesId", 0),
                name,
            ))
        self._fill_paths(items, "episode")
        return items

    def _fill_paths(self, items: list[WantedItem], kind: str) -> None:
        """Resolve file paths in batches - the wanted endpoints don't return them."""
        endpoint, param = ("/movies", "radarrid[]") if kind == "movie" else ("/episodes", "episodeid[]")
        ids = [i.item_id for i in items if i.item_id]
        by_id = {i.item_id: i for i in items}
        for chunk in (ids[n:n + 50] for n in range(0, len(ids), 50)):
            data = self._get(endpoint, [(param, str(i)) for i in chunk])
            for row in data.get("data", []):
                key = row.get("radarrId") if kind == "movie" else row.get("sonarrEpisodeId")
                item = by_id.get(key)
                if item and row.get("path"):
                    item.path = self.cfg.to_local(row["path"])

    # -- write-back -------------------------------------------------------

    def rescan(self, kind: str, series_id: int, item_id: int) -> bool:
        """Ask Bazarr to re-index the folder so the new sidecar is picked up."""
        if not (self.enabled and self.cfg.notify_bazarr):
            return False
        if kind == "movie" and item_id:
            return self._patch("/movies", {"radarrid": item_id, "action": "scan-disk"})
        if kind == "episode" and series_id:
            return self._patch("/series", {"seriesid": series_id, "action": "scan-disk"})
        return False

    # -- transport --------------------------------------------------------

    def _get(self, path: str, params) -> dict:
        try:
            response = self._client.get(path, params=params)
        except httpx.RequestError as exc:
            log.warning("bazarr unreachable (%s): %s", path, exc)
            return {}
        if response.status_code != 200:
            log.warning("bazarr %s returned %s", path, response.status_code)
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def _patch(self, path: str, params: dict) -> bool:
        try:
            response = self._client.patch(path, params=params)
        except httpx.RequestError as exc:
            log.warning("bazarr rescan failed: %s", exc)
            return False
        return response.status_code < 300


def _wants(row: dict, lang: str) -> bool:
    for missing in row.get("missing_subtitles") or []:
        if isinstance(missing, dict):
            if missing.get("code2") == lang or missing.get("code3") == lang:
                return True
        elif isinstance(missing, str) and lang in missing:
            return True
    return False
