#!/usr/bin/env python3
"""
preprocess.py
=============

Cut long recordings into clean, ready‑to‑mix *bits*.

Pipeline
--------
1.  **Load** every WAV in ``in/``  
    – mono‑down‑mix if stereo  
    – resample to **16 kHz** (fast, speech‑friendly)
2.  **Noise‑reduce** the entire file with *noisereduce* (light spectral gate)
3.  **Silence‑split** – any ≥150 ms gap below (loudness −14 dB) ends a segment
4.  **Chop** each segment into ≤2 s windows (50 % hop)
5.  **Pitch‑analyse** → optional **safe shift** (±4 st male↔female) via *PyWorld*
6.  **Fades**:  ≤1 s ⇒ 100 ms in + out • >1 s ⇒ 300 ms in + out
7.  **Export** bits to ``bits/`` + write `debug.csv`

Run
---
::

    python preprocess.py     # slices every WAV in ./in/

All output lands in **bits/**; originals are untouched.
"""
# ── imports ──────────────────────────────────────────────────────────────
import os, csv, time, random
import numpy as np, librosa, noisereduce as nr
import scipy.signal as ss
from scipy.io import wavfile
from pydub import AudioSegment, silence
import pyworld as pw  # formant‑preserving pitch‑shift

# ── folders ──────────────────────────────────────────────────────────────
IN_DIR, BITS_DIR = "in", "bits"
os.makedirs(BITS_DIR, exist_ok=True)
CSV_PATH, WRITE_CSV = "debug.csv", True

# ── processing constants ────────────────────────────────────────────────
MIN_SIL_MS, KEEP_SIL_MS = 150, 100  # silence‑split params
WIN_MS = 2000  # chopping window (ms)
MIN_MS, MAX_MS = 300, 2000  # keep only within this range
FADE_S, FADE_L, EDGE_MS = 100, 300, 1000  # fade ≤1 s / >1 s threshold

MALE, FEM = (80, 150), (175, 300)  # safe pitch ranges (Hz)
MAX_SHIFT_ST = 4  # ±4 st cap


# ── helpers ──────────────────────────────────────────────────────────────
def to_i16(arr: np.ndarray) -> np.ndarray:
    """Return little‑endian int16, normalised if float."""
    if arr.dtype != np.int16:
        peak = np.max(np.abs(arr)) or 1.0
        arr = (arr / peak * 32767).astype("<i2")
    return arr


def fade_np(buf: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Linear in/out fade in NumPy domain."""
    n = int(sr * ms / 1000)
    ramp = np.linspace(0, 1, n, False)
    buf = buf.astype(np.float32)
    buf[:n] *= ramp
    buf[-n:] *= ramp[::-1]
    return to_i16(buf)


def yin_pitch(x: np.ndarray, sr: int):
    f = librosa.yin(x.astype(np.float32), 50, 1000, sr)
    f = f[f > 0]
    return float(np.median(f)) if f.size else None


def safe_shift(base: float | None) -> float:
    if not base:
        return 0.0
    if MALE[0] <= base <= MALE[1]:
        tgt = random.uniform(*FEM)
    elif FEM[0] <= base <= FEM[1]:
        tgt = random.uniform(*MALE)
    else:
        return 0.0
    return float(np.clip(12 * np.log2(tgt / base), -MAX_SHIFT_ST, MAX_SHIFT_ST))


def world_shift(arr: np.ndarray, sr: int, st: float) -> np.ndarray:
    if st == 0:
        return arr
    x = arr.astype(np.float64)
    f0, t = pw.harvest(x, sr)
    f0 *= 2 ** (st / 12)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    y = pw.synthesize(f0, sp, ap, sr)
    return to_i16(np.clip(y, -1, 1))


def chop(seg: AudioSegment):
    """Yield (subchunk, offset_ms) windows of ≤WIN_MS with 50 % hop."""
    i = 0
    while i < len(seg):
        part = seg[i : i + WIN_MS]
        if len(part) >= MIN_MS:
            yield part, i
        i += WIN_MS // 2


# ── main ─────────────────────────────────────────────────────────────────
bit_idx, rows, t0 = 0, [], time.time()
for idx, fname in enumerate(sorted(os.listdir(IN_DIR)), 1):
    if not fname.lower().endswith(".wav"):
        continue
    print(f"[{idx}] {fname}")

    # 1 – load + mono + 16 kHz -------------------------------------------
    sr, data = wavfile.read(os.path.join(IN_DIR, fname))
    if data.ndim == 2:
        data = data.mean(1).astype(np.int16)
    if sr > 16_000:
        dec = sr // 16_000
        data = ss.decimate(data, dec, ftype="fir", zero_phase=True)
        sr //= dec
        data = to_i16(data)

    # 2 – noise‑reduction -----------------------------------------------
    data = nr.reduce_noise(y=data, sr=sr)
    wavfile.write("tmp.wav", sr, to_i16(data))
    full = AudioSegment.from_wav("tmp.wav")
    os.remove("tmp.wav")

    # 3 – split on silence ----------------------------------------------
    segs = silence.split_on_silence(full, MIN_SIL_MS, full.dBFS - 14, KEEP_SIL_MS)
    print(f"  → {len(segs)} segments")

    # 4 – iterate windows -----------------------------------------------
    for si, seg in enumerate(segs, 1):
        for ci, (sub, off) in enumerate(chop(seg), 1):
            arr = np.array(sub.get_array_of_samples()).astype(np.float32) / 32768

            base = yin_pitch(arr, sr)
            st = safe_shift(base)

            tried, final, fp, used = [], None, None, 0.0
            if st:
                for f in (1.0, 0.75, 0.5):
                    s = st * f
                    tried.append(round(s, 2))
                    y = world_shift(to_i16(arr), sr, s) / 32768
                    p = yin_pitch(y, sr)
                    if p:
                        final, fp, used = y, p, s
                        break
                else:
                    final, fp = arr, base
                    tried.append(0.0)
            else:
                final, fp = arr, base
                tried.append(0.0)

            dur = len(final) / sr * 1000
            if not (MIN_MS <= dur <= MAX_MS):
                continue
            fade = FADE_S if dur <= EDGE_MS else FADE_L
            final = fade_np(to_i16(final * 32768), sr, fade)
            dur = len(final) / sr * 1000

            bit = f"bit_{bit_idx:04d}.wav"
            wavfile.write(os.path.join(BITS_DIR, bit), sr, final)

            rows.append(
                {
                    "bit_file": bit,
                    "original_file": fname,
                    "start_ms": off,
                    "duration_ms": round(dur),
                    "base_pitch_Hz": round(base, 1) if base else None,
                    "shift_semitones": round(used, 2),
                    "final_pitch_Hz": round(fp, 1) if fp else None,
                    "shift_tried": "/".join(map(str, tried)),
                }
            )

            print(
                f"    [{si}.{ci}] {bit} | {round(dur)} ms | {base or '?'} Hz → {used:+.2f} st → {fp or '?'} Hz"
            )
            bit_idx += 1

print(f"\n✔ {bit_idx} bits written in {time.time()-t0:.1f} s")

# CSV log -------------------------------------------------------------
if WRITE_CSV and rows:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"CSV → {CSV_PATH}")
