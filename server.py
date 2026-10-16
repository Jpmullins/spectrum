#!/usr/bin/env python3
"""A spectrum viewer for torchsig.spectrum: waterfall, PSD and truth over a live synthetic band.

Run it on the machine that generates (spark01), view it from anywhere over an SSH tunnel::

    source ~/.venvs/siggpt/bin/activate
    python /home/jpmullins/code/rfml/spectrum/server.py --port 8080
    # then, on your laptop:
    ssh -L 8080:127.0.0.1:8080 spark01
    # and open http://127.0.0.1:8080

Nothing is installed: this uses the standard library's HTTP server, plus numpy and torch, which the siggpt
environment already has. The generator runs about fifteen times slower than real time, so the viewer polls
rather than pushing, and reports the ratio it is achieving.

How it fits together:

* A worker thread pulls chunks from a :class:`torchsig.spectrum.BandStream`, turns each into spectrogram rows
  on the GPU, and appends them to a ring buffer with the truth of what was on the air.
* The browser polls ``/api/rows?since=<seq>`` for new rows as quantized bytes, and draws the waterfall and
  the power spectrum itself.
* ``/api/control`` reconfigures the stream, which restarts the worker on a fresh :class:`BandStream`.

Stream time is what the waterfall shows, and it advances only as fast as the band can be generated. The
samples are exactly what a receiver would have delivered; only the wall clock differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchsig.spectrum import BandStream, Receiver, named_band
from torchsig.spectrum.catalog import load_catalog
from torchsig.spectrum.population import Environment

LOG = logging.getLogger("spectrum")
STATIC = Path(__file__).resolve().parent / "static"

# The waterfall is sent as bytes: each row is `n_bins` values spanning DB_FLOOR..DB_CEIL.
# The displayed window. The thermal floor is about -117 dBm in a 97.7 kHz bin at 100 MS/s, and the loudest
# emitters reach about -60, so this spans the noise to the strongest signal with a little headroom either way.
DB_FLOOR = -130.0
DB_CEIL = -55.0
ROW_HISTORY = 4096  # rows kept for clients that fall behind


def _db_to_u8(db: np.ndarray) -> np.ndarray:
    scaled = (db - DB_FLOOR) * (255.0 / (DB_CEIL - DB_FLOOR))
    return np.clip(scaled, 0, 255).astype(np.uint8)


class Engine:
    """Generates the band and reduces it to spectrogram rows, in a worker thread."""

    def __init__(self, **settings: Any) -> None:
        self.lock = threading.Lock()
        self.rows: deque[tuple[int, np.ndarray]] = deque(maxlen=ROW_HISTORY)
        self.sequence = 0
        self.truth: deque[dict[str, Any]] = deque()  # pruned by time to match the row history
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.stream_time_s = 0.0
        self.wall_time_s = 0.0
        self.chunks_done = 0
        self.psd_db: np.ndarray | None = None
        self.peak_db: np.ndarray | None = None
        self.settings = {
            "band": "ism_2450",
            "clutter": "urban_dense",
            "hour_of_day": 20.0,
            "seed": 7,
            "tune_mhz": 2441.75,
            "decimation": 2,
            "gain_db": 20.0,
            "lo_offset_mhz": 0.0,
            "image_rejection_db": 60.0,
            "dc_spur_dbfs": -80.0,
            "nfft": 1024,
            "row_ms": 0.5,
            "paused": False,
        }
        self.settings.update(settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window: torch.Tensor | None = None
        self.stream: BandStream | None = None
        # A row is a whole number of FFT frames, so its duration is quantized and is rarely exactly the
        # requested row_ms. Everything that converts a row index to a time must use the realized value, or
        # the waterfall's time axis and the truth overlay drift apart as the stream runs.
        self.row_duration_s = 0.0
        self.restart()

    # --- configuration --------------------------------------------------- #

    def _build(self) -> BandStream:
        s = self.settings
        receiver = Receiver(
            tune_hz=float(s["tune_mhz"]) * 1e6,
            decimation=int(s["decimation"]),
            gain_db=float(s["gain_db"]),
            lo_offset_hz=float(s["lo_offset_mhz"]) * 1e6,
            image_rejection_db=float(s["image_rejection_db"]),
            dc_spur_dbfs=float(s["dc_spur_dbfs"]),
        )
        plan = named_band(str(s["band"]))
        return BandStream(
            plan,
            receiver,
            seed=int(s["seed"]),
            environment=Environment(clutter=str(s["clutter"]), hour_of_day=float(s["hour_of_day"])),
            chunk_s=0.01,
        )

    def restart(self) -> None:
        """Rebuild the stream and restart the worker. Called on any setting change."""
        self.stop()
        with self.lock:
            self.stream = self._build()
            self.rows.clear()
            self.truth.clear()
            self.sequence = 0
            self.stream_time_s = 0.0
            self.wall_time_s = 0.0
            self.chunks_done = 0
            self.psd_db = None
            self.peak_db = None
            nfft = int(self.settings["nfft"])
            self.window = torch.hann_window(nfft, periodic=False, device=self.device, dtype=torch.float32)
            row_samples = max(int(round(float(self.settings["row_ms"]) * 1e-3 * self.stream.sample_rate_hz)), nfft)
            frames_per_row = max(row_samples // nfft, 1)
            self.row_duration_s = frames_per_row * nfft / self.stream.sample_rate_hz
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="band-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            self.thread = None

    def update(self, changes: dict[str, Any]) -> list[str]:
        """Apply setting changes. Returns the names that changed; restarts if any need it."""
        changed = []
        structural = {"band", "clutter", "hour_of_day", "seed", "tune_mhz", "decimation", "gain_db",
                      "lo_offset_mhz", "image_rejection_db", "dc_spur_dbfs", "nfft", "row_ms"}
        for key, value in changes.items():
            if key not in self.settings:
                continue
            if self.settings[key] != value:
                self.settings[key] = value
                changed.append(key)
        if any(k in structural for k in changed):
            self.restart()
        return changed

    # --- the worker ------------------------------------------------------ #

    def _run(self) -> None:
        assert self.stream is not None
        stream = self.stream
        nfft = int(self.settings["nfft"])
        frames_per_row = max(int(round(self.row_duration_s * stream.sample_rate_hz)) // nfft, 1)
        started = time.perf_counter()
        index = 0
        while not self.stop_event.is_set():
            if self.settings["paused"]:
                time.sleep(0.05)
                started = time.perf_counter() - self.wall_time_s
                continue
            try:
                chunk = stream.chunk(index)
            except Exception:  # pragma: no cover - a generator fault should not kill the viewer
                LOG.exception("chunk %d failed", index)
                time.sleep(0.5)
                index += 1
                continue
            rows, psd = self._spectrogram(chunk.sc16, nfft, frames_per_row)
            with self.lock:
                for row in rows:
                    self.rows.append((self.sequence, row))
                    self.sequence += 1
                for burst in chunk.bursts_overlapping():
                    self.truth.append(
                        {
                            "t": round(burst.start_s, 6),
                            "d": round(burst.duration_s, 6),
                            "f": round(burst.center_hz / 1e6, 4),
                            "bw": round(burst.bandwidth_hz / 1e6, 4),
                            "tech": burst.technology_id,
                            "kind": burst.kind,
                            "p": round(burst.received_power_dbm, 1),
                        }
                    )
                # Keep truth for exactly as long as the rows it annotates, so an old row still has its
                # overlay and the buffer cannot grow without bound.
                oldest_row_t = max(0.0, (self.sequence - len(self.rows)) * self.row_duration_s)
                while self.truth and self.truth[0]["t"] + self.truth[0]["d"] < oldest_row_t:
                    self.truth.popleft()
                self.psd_db = psd
                self.peak_db = psd if self.peak_db is None else np.maximum(self.peak_db, psd)
                self.stream_time_s = (index + 1) * stream.chunk_s
                self.wall_time_s = time.perf_counter() - started
                self.chunks_done = index + 1
            index += 1

    def _spectrogram(self, sc16: np.ndarray, nfft: int, frames_per_row: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Spectrogram rows and the chunk's mean power spectrum, both in dBm per bin."""
        assert self.stream is not None and self.window is not None
        receiver = self.stream.receiver
        # sc16 -> complex64 at the antenna, in sqrt(mW), on the device.
        codes = torch.from_numpy(np.ascontiguousarray(sc16)).to(self.device, torch.float32)
        codes = codes / float(1 << (16 - receiver.adc_bits))
        levels = float(2 ** (receiver.adc_bits - 1))
        gain = 10.0 ** (receiver.gain_db / 20.0)
        full_scale = math.sqrt(10.0 ** (receiver.adc_full_scale_dbm / 10.0))
        rails = codes / levels * full_scale / gain
        x = torch.complex(rails[:, 0], rails[:, 1])

        n_frames = x.numel() // nfft
        n_rows = max(n_frames // frames_per_row, 1)
        usable = n_rows * frames_per_row * nfft
        frames = x[:usable].reshape(-1, nfft) * self.window
        spectrum = torch.fft.fftshift(torch.fft.fft(frames, dim=-1), dim=-1)
        # Scale so the bins sum to the total power (Parseval). For a windowed FFT,
        #     sum_k |X_k|^2 = nfft * sum_n |x_n w_n|^2 = nfft^2 * mean(|x|^2) * mean(w^2),
        # so dividing by nfft^2 * mean(w^2) makes the bins sum to mean(|x|^2). Dividing by only
        # nfft * mean(w^2) would make every bin read the *total* band power, which is 10*log10(nfft)
        # = 30 dB too hot at a 1024-point FFT.
        norm = (nfft**2) * float(torch.mean(self.window**2))
        power = (spectrum.real**2 + spectrum.imag**2) / norm
        power = power.reshape(n_rows, frames_per_row, nfft).mean(dim=1)
        db = 10.0 * torch.log10(power + 1e-20)
        db_np = db.cpu().numpy()
        psd = 10.0 * np.log10(power.mean(dim=0).cpu().numpy() + 1e-20)
        return [_db_to_u8(r) for r in db_np], psd.astype(np.float32)

    # --- what the client reads ------------------------------------------- #

    def config(self) -> dict[str, Any]:
        assert self.stream is not None
        stream = self.stream
        receiver = stream.receiver
        nfft = int(self.settings["nfft"])
        catalog = load_catalog()
        # Only plans a viewer can read: a channel grid of a few tens of lines means something, a 100 kHz
        # carrier raster is just a grid. Bluetooth LE is shown by its three advertising channels, which is
        # where its energy actually concentrates, not by all forty.
        markers = []
        for technology in stream.plan.technologies:
            centers = technology.channels_in(receiver.f_low_hz, receiver.f_high_hz)
            if not centers:
                continue
            label = technology.id
            if technology.id == "bluetooth_le":
                centers = tuple(c for c in (2402e6, 2426e6, 2480e6) if receiver.f_low_hz <= c <= receiver.f_high_hz)
                label = "bluetooth_le (advertising)"
            elif len(centers) > 20:
                continue
            markers.append(
                {
                    "tech": technology.id,
                    "name": technology.name,
                    "label": label,
                    "bw_mhz": (technology.occupied_bandwidth_hz or 0.0) / 1e6,
                    "centers_mhz": [round(c / 1e6, 4) for c in centers],
                }
            )
        counts: dict[str, int] = {}
        for emitter in stream.audible_emitters:
            counts[emitter.technology_id] = counts.get(emitter.technology_id, 0) + 1
        return {
            "settings": self.settings,
            "device": str(self.device),
            "nfft": nfft,
            "n_bins": nfft,
            "f_low_mhz": receiver.f_low_hz / 1e6,
            "f_high_mhz": receiver.f_high_hz / 1e6,
            "tune_mhz": receiver.tune_hz / 1e6,
            "sample_rate_msps": receiver.sample_rate_hz / 1e6,
            "bin_khz": receiver.sample_rate_hz / nfft / 1e3,
            "row_ms": self.row_duration_s * 1e3,  # realized, not requested: a row is a whole number of frames
            "row_ms_requested": float(self.settings["row_ms"]),
            "db_floor": DB_FLOOR,
            "db_ceil": DB_CEIL,
            "noise_power_dbm": receiver.noise_power_dbm,
            "noise_per_bin_dbm": receiver.noise_power_dbm_in(receiver.sample_rate_hz / nfft),
            "dc_spur_mhz": receiver.dc_spur_hz / 1e6,
            "dc_spur_above_noise_db": receiver.dc_spur_above_noise_db(receiver.sample_rate_hz / nfft),
            "dc_spur_escapes_band": receiver.dc_spur_escapes_band,
            "max_lo_offset_mhz": receiver.max_lo_offset_hz / 1e6,
            "receiver": receiver.describe(),
            "provenance": stream.provenance(),
            "emitter_counts": counts,
            "markers": markers,
            "clutter_classes": ["urban_dense", "urban", "suburban", "rural", "open", "indoor"],
            "bands": ["ism_2450", "ism_915", "uhf_lmr"],
            "technologies": {t.id: {"name": t.name, "tier": t.tier} for t in stream.plan.technologies},
            "catalog_size": len(catalog),
        }

    def stats(self) -> dict[str, Any]:
        with self.lock:
            ratio = (self.stream_time_s / self.wall_time_s) if self.wall_time_s > 0 else 0.0
            return {
                "stream_time_s": round(self.stream_time_s, 4),
                "wall_time_s": round(self.wall_time_s, 2),
                "realtime_ratio": round(ratio, 4),
                "chunks": self.chunks_done,
                "sequence": self.sequence,
                "rows_buffered": len(self.rows),
                "paused": bool(self.settings["paused"]),
            }

    def rows_since(self, since: int, limit: int = 256) -> tuple[int, int, bytes, list[dict[str, Any]]]:
        """Rows with sequence >= ``since``, as raw bytes, plus the truth that overlaps them."""
        with self.lock:
            selected = [(seq, row) for seq, row in self.rows if seq >= since][:limit]
            if not selected:
                return since, 0, b"", []
            first = selected[0][0]
            blob = b"".join(row.tobytes() for _, row in selected)
            row_s = self.row_duration_s
            t_lo = first * row_s
            t_hi = (first + len(selected)) * row_s
            truth = [b for b in self.truth if b["t"] < t_hi and t_lo < b["t"] + b["d"]]
            return first, len(selected), blob, truth

    def spectrum(self) -> dict[str, Any]:
        with self.lock:
            if self.psd_db is None:
                return {"psd": [], "peak": []}
            return {
                "psd": [round(float(v), 2) for v in self.psd_db],
                "peak": [round(float(v), 2) for v in (self.peak_db if self.peak_db is not None else self.psd_db)],
            }

    def reset_peak(self) -> None:
        with self.lock:
            self.peak_db = None

    def emitters(self) -> list[dict[str, Any]]:
        assert self.stream is not None
        receiver = self.stream.receiver
        out = []
        for e in sorted(self.stream.audible_emitters, key=lambda e: -e.received_power_dbm):
            out.append(
                {
                    "id": e.id,
                    "tech": e.technology_id,
                    "f_mhz": round(e.center_hz / 1e6, 4),
                    "bw_mhz": round(e.occupied_bandwidth_hz / 1e6, 4),
                    "eirp_dbm": round(e.eirp_dbm, 1),
                    "range_m": round(e.range_m, 1),
                    "rx_dbm": round(e.received_power_dbm, 1),
                    "snr_db": round(receiver.snr_db(e.received_power_dbm, max(e.occupied_bandwidth_hz, 1.0)), 1),
                    "activity": e.activity.get("kind", "?"),
                }
            )
        return out


class Handler(BaseHTTPRequestHandler):
    engine: Engine  # set on the server class

    protocol_version = "HTTP/1.1"
    server_version = "spectrum/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        LOG.debug(fmt, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(payload).encode(), "application/json", status)

    def do_GET(self) -> None:  # noqa: N802 - the stdlib's spelling
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        engine = type(self).engine
        try:
            if path in ("/", "/index.html"):
                self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._json(engine.config())
            elif path == "/api/stats":
                self._json(engine.stats())
            elif path == "/api/spectrum":
                self._json(engine.spectrum())
            elif path == "/api/emitters":
                self._json(engine.emitters())
            elif path == "/api/rows":
                limit = min(int(params.get("limit", 256)), 1024)
                since = int(params.get("since", 0))
                if since < 0:  # "give me the newest", for a client that has just connected
                    since = max(0, engine.stats()["sequence"] - limit)
                first, count, blob, truth = engine.rows_since(since, limit)
                header = json.dumps(
                    {"first": first, "count": count, "n_bins": int(engine.settings["nfft"]), "truth": truth}
                ).encode()
                body = struct.pack("<I", len(header)) + header + blob
                self._send(body, "application/octet-stream")
            else:
                self._json({"error": "not found", "path": path}, 404)
        except BrokenPipeError:  # pragma: no cover - a client that navigated away
            pass
        except Exception as exc:  # pragma: no cover
            LOG.exception("GET %s failed", self.path)
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        engine = type(self).engine
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._json({"error": f"bad JSON: {exc}"}, 400)
            return
        if self.path == "/api/control":
            try:
                changed = engine.update(payload)
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"changed": changed, "config": engine.config()})
        elif self.path == "/api/reset_peak":
            engine.reset_peak()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback, for SSH tunnels)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--band", default="ism_2450")
    parser.add_argument("--clutter", default="urban_dense")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tune-mhz", type=float, default=2441.75)
    parser.add_argument("--decimation", type=int, default=2)
    parser.add_argument("--gain-db", type=float, default=20.0)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--row-ms", type=float, default=0.5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    engine = Engine(
        band=args.band,
        clutter=args.clutter,
        seed=args.seed,
        tune_mhz=args.tune_mhz,
        decimation=args.decimation,
        gain_db=args.gain_db,
        nfft=args.nfft,
        row_ms=args.row_ms,
    )
    Handler.engine = engine
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        engine.stop()
        raise SystemExit(
            f"cannot bind {args.host}:{args.port} ({exc}). Another viewer is probably still running: "
            f"find it with `ss -ltnp | grep {args.port}` and stop it, or pass a different --port."
        ) from exc
    config = engine.config()
    LOG.info(
        "band %s: %.2f-%.2f MHz at %.1f MS/s, %d audible emitters, FFT on %s",
        args.band,
        config["f_low_mhz"],
        config["f_high_mhz"],
        config["sample_rate_msps"],
        sum(config["emitter_counts"].values()),
        config["device"],
    )
    LOG.info("serving on http://%s:%d  (ssh -L %d:127.0.0.1:%d %s)", args.host, args.port, args.port, args.port, "spark01")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("stopping")
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
