#!/usr/bin/env python3
"""
collage.py – assemble N stereo tracks of random “bits” on a beat grid.

Energy symbols         (probability of spawning a bit *per beat*)
  S  0.01   L 0.3   M 0.6   H 0.9
  l  0.3    m 0.6   h 0.9   (solo: only track-0 active)

C / D   maintain probability but ramp gain   (±6 dB across section)

Usage
-----
python collage.py [--pattern STR] [--bpm N] [--apply-styles] [--csv]

  --pattern       e.g. "SSCLLMMhhDD". Default "M".
  --bpm           beats-per-minute (default 120).
  --apply-styles  enable pan / echo / reverb / stretch / ±gain.
  --csv           write collage_debug.csv.
  --help          this text.

Outputs
-------
out/collage_YY.MM.DD_HH.MM_[s|ns]/
    track0.wav … track9.wav   master.wav   pattern.txt   [debug.csv]
"""
# ─── imports ────────────────────────────────────────────────────────────────
import os, sys, csv, random, datetime, math
import numpy as np
import librosa, noisereduce as nr
from pydub import AudioSegment, effects

# ─── constants ──────────────────────────────────────────────────────────────
BITS_DIR, TRACK_COUNT, TARGET_MS = "bits", 5, 40_000
FADE_MS = 100
REVERB_DELAYS, REVERB_DECAY = [120, 150, 170], 0.2
ECHO_CHANCE, ECHO_DELAY_MS = 0.5, 800
STRETCH_PROB, STRETCH_RANGE = 0.3, (0.8, 1.2)
PAN_RANGE, GAIN_RANGE_DB = (-1, 1), (-3, +3)
SAMPLE_RATE, CHANNELS = 44_100, 2

PROB_TABLE = {
    "S": 0.01,
    "L": 0.2,
    "M": 0.5,
    "H": 0.9,
    "l": 0.4,
    "m": 0.7,
    "h": 0.99,
    "C": None,
    "D": None,
}

# ─── help flag --------------------------------------------------------------
if "--help" in sys.argv:
    print(__doc__)
    sys.exit(0)


# ─── CLI --------------------------------------------------------------------
def argval(flag, default):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


pattern = argval("--pattern", "M")
bpm = int(argval("--bpm", 120))
apply_styles = "--apply-styles" in sys.argv
write_csv = "--csv" in sys.argv

for ch in pattern:
    if ch not in PROB_TABLE:
        sys.exit(f"[error] unknown pattern symbol '{ch}'")

beat_ms = int(60_000 / bpm)
total_beats = TARGET_MS // beat_ms
pattern_syms = list(pattern)
sections = len(pattern_syms)
beats_per_sec = total_beats // sections  # integer; last sec may run long
sec_bounds = [b * beats_per_sec for b in range(sections)] + [total_beats]

# ─── output folder ----------------------------------------------------------
stamp = datetime.datetime.now().strftime("collage_%y.%m.%d_%H.%M_") + (
    "s" if apply_styles else "ns"
)
OUT_DIR = os.path.join("out", stamp)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"⇒ Output folder: {OUT_DIR}   (beat = {beat_ms} ms)\n")
with open(os.path.join(OUT_DIR, "pattern.txt"), "w") as f:
    f.write(pattern + "\n")

# ─── load bits --------------------------------------------------------------
bits = [
    (bf, AudioSegment.from_wav(os.path.join(BITS_DIR, bf)))
    for bf in os.listdir(BITS_DIR)
    if bf.lower().endswith(".wav")
]
print(f"{len(bits)} bits loaded\n")


# ─── helper FX --------------------------------------------------------------
def apply_reverb(seg):
    out = seg
    for i, d in enumerate(REVERB_DELAYS, 1):
        out = out.overlay(seg - (i * (1 - REVERB_DECAY) * 10), position=d)
    return out


def apply_echo(seg):
    return seg.overlay(seg - 10, position=ECHO_DELAY_MS)


def maybe_stretch(seg):
    if random.random() > STRETCH_PROB:
        return seg
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768
    y2 = librosa.effects.time_stretch(y=y, rate=random.uniform(*STRETCH_RANGE))
    return AudioSegment(
        (y2 * 32767).astype(np.int16).tobytes(),
        frame_rate=seg.frame_rate,
        sample_width=2,
        channels=1,
    )


def auto_fix(seg):
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768
    sr = seg.frame_rate
    rms = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec = np.abs(np.fft.rfft(y))
    hi = spec[int(len(spec) * 0.7) :].mean()
    lo = spec[: int(len(spec) * 0.1)].mean()
    out = seg
    if rms > -30:
        out = AudioSegment(
            (nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8) * 32767)
            .astype(np.int16)
            .tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
    elif cent < 1_000:
        out = (
            out.high_pass_filter(40)
            .low_pass_filter(16_000)
            .apply_gain(+2)
            .apply_gain(-2)
        )
    if hi > lo * 2:
        out = effects.compress_dynamic_range(
            out, threshold=-18, ratio=2, attack=10, release=250
        ).apply_gain(+2)
    return out


# ─── create blank stereo tracks --------------------------------------------
blank = AudioSegment.silent(duration=TARGET_MS, frame_rate=SAMPLE_RATE).set_channels(
    CHANNELS
)
tracks = [blank[:] for _ in range(TRACK_COUNT)]

# ─── iterate beats ----------------------------------------------------------
print("Building tracks …")
metadata = []
gain_start = 0.0
sec_idx = 0
for beat in range(total_beats):
    if beat == sec_bounds[sec_idx + 1]:
        # section finished – jump to next
        gain_start = gain_end
        sec_idx += 1
    sym = pattern_syms[sec_idx]
    prob = PROB_TABLE[sym] if PROB_TABLE[sym] is not None else 0.5
    active = [True] * TRACK_COUNT
    if sym in "lmh":
        active = [t == 0 for t in range(TRACK_COUNT)]

    # gain ramp inside current section
    pos_ratio = (beat - sec_bounds[sec_idx]) / max(
        1, sec_bounds[sec_idx + 1] - sec_bounds[sec_idx]
    )
    if sym == "C":
        gain_end = +6.0
    elif sym == "D":
        gain_end = -6.0
    else:
        gain_end = gain_start
    beat_gain = gain_start + (gain_end - gain_start) * pos_ratio

    pos_ms = beat * beat_ms

    for t in range(TRACK_COUNT):
        if not active[t] or random.random() > prob:
            continue
        bf, seg = random.choice(bits)
        seg = auto_fix(seg)
        seg = effects.normalize(seg, headroom=6.0)
        if apply_styles:
            seg = maybe_stretch(seg).fade_in(FADE_MS).fade_out(FADE_MS)
            if random.random() < ECHO_CHANCE:
                seg = apply_echo(seg)
            seg = apply_reverb(seg)
            seg = seg.pan(random.uniform(*PAN_RANGE))
            seg = seg.apply_gain(random.uniform(*GAIN_RANGE_DB) + beat_gain)
        else:
            seg = seg.apply_gain(beat_gain)
        tracks[t] = tracks[t].overlay(seg, position=pos_ms)
        metadata.append(
            {
                "track": t,
                "bit": bf,
                "beat": beat,
                "start_ms": pos_ms,
                "dur_ms": len(seg),
            }
        )

# ─── export tracks ----------------------------------------------------------
PAD_DB = -12 * math.log10(TRACK_COUNT)  # ≈ –10 dB with 10 tracks
for i, tr in enumerate(tracks):
    path = os.path.join(OUT_DIR, f"track{i}.wav")
    tr.export(path, format="wav")
    print(f"  track {i} → {path}")

# ─── master ----------------------------------------------------------------
master = blank[:]  # silence
for tr in tracks:
    master = master.overlay(tr.apply_gain(PAD_DB))
# master = master.apply_gain(+6)
m_path = os.path.join(OUT_DIR, "master.wav")
master.export(m_path, format="wav")
print(f"\n✔ master → {m_path}")

# ─── optional CSV ----------------------------------------------------------
if write_csv and metadata:
    csv_p = os.path.join(OUT_DIR, "collage_debug.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metadata[0].keys())
        w.writeheader()
        w.writerows(metadata)
    print(f"✔ CSV    → {csv_p}")

print("\nDone.")
