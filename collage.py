#!/usr/bin/env python3
"""
collage.py  –  Build N stereo tracks from random speech bits,
               then sum them into master.wav.

    python collage.py                      # dry mix, no CSV
    python collage.py --apply-styles       # pan/echo/reverb/stretch/gain
    python collage.py --apply-styles --csv # same + placement CSV
"""
import os, sys, random, csv, datetime
import numpy as np
import librosa, noisereduce as nr
from pydub import AudioSegment, effects
from pydub.generators import Sine

# ─── USER CONFIG ───────────────────────────────────────────────────────────
BITS_DIR = "bits"
TRACK_COUNT = 10  # how many parallel tracks
TARGET_MS = 30_000  # length of each track

# silence between bits (no overlap within a track)
MIN_SIL_MS = 100
MAX_SIL_MS = 3_000

# FX (used only with --apply-styles)
FADE_MS = 100
REVERB_DELAYS = [120, 150, 170]
REVERB_DECAY = 0.1
ECHO_CHANCE = 0.3
ECHO_DELAY_MS = 500
STRETCH_PROB = 0.3
STRETCH_RANGE = (0.8, 1.2)
PAN_RANGE = (-1.0, 1.0)
GAIN_RANGE_DB = (-3, +3)

# hum bed
HUM_FREQ_HZ = 60
HUM_GAIN_DB = -10

# audio template
SAMPLE_RATE = 44_100
CHANNELS = 2
# ───────────────────────────────────────────────────────────────────────────

# --- command-line flags ----------------------------------------------------
apply_styles = "--apply-styles" in sys.argv
write_csv = "--csv" in sys.argv

# --- timestamped output folder --------------------------------------------
now = datetime.datetime.now()
stamp = now.strftime("collage_%y.%m.%d_%H.%M_") + ("s" if apply_styles else "ns")
OUT_DIR = os.path.join("out", stamp)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"⇒ Output folder:  {OUT_DIR}\n")

# --- load bits -------------------------------------------------------------
bit_files = [f for f in os.listdir(BITS_DIR) if f.lower().endswith(".wav")]
bits = [(bf, AudioSegment.from_wav(os.path.join(BITS_DIR, bf))) for bf in bit_files]
print(f"{len(bits)} bits loaded from '{BITS_DIR}'\n")


# --- helper FX -------------------------------------------------------------
def apply_reverb(seg):
    out = seg
    for i, d in enumerate(REVERB_DELAYS, 1):
        tap = seg - (i * (1 - REVERB_DECAY) * 10)
        out = out.overlay(tap, position=d)
    return out


def apply_echo(seg):
    return seg.overlay(seg - 10, position=ECHO_DELAY_MS)


def maybe_stretch(seg):
    if random.random() > STRETCH_PROB:
        return seg
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768
    factor = random.uniform(*STRETCH_RANGE)
    y2 = librosa.effects.time_stretch(y=y, rate=factor)
    return AudioSegment(
        (y2 * 32767).astype(np.int16).tobytes(),
        frame_rate=seg.frame_rate,
        sample_width=2,
        channels=1,
    )


def auto_fix(seg):
    """Very light denoise/EQ/compress per bit."""
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768
    sr = seg.frame_rate
    rms = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec = np.abs(np.fft.rfft(y))
    hi, lo = spec[int(len(spec) * 0.7) :].mean(), spec[: int(len(spec) * 0.1)].mean()
    out = seg
    if rms > -30:  # noisy
        y_dn = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        out = AudioSegment(
            (y_dn * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
    elif cent < 1_000:  # dull
        out = (
            out.high_pass_filter(40)
            .low_pass_filter(16_000)
            .apply_gain(+2)
            .apply_gain(-2)
        )
    if hi > lo * 2:  # music-heavy
        out = effects.compress_dynamic_range(
            out, threshold=-18, ratio=2, attack=10, release=250
        ).apply_gain(+2)
    return out


# ---------------------------------------------------------------------------

# --- blank stereo template -------------------------------------------------
blank = AudioSegment.silent(duration=TARGET_MS, frame_rate=SAMPLE_RATE).set_channels(
    CHANNELS
)
tracks = [blank[:] for _ in range(TRACK_COUNT)]
metadata = []

print("Building tracks …")
for t in range(TRACK_COUNT):
    cursor = 0
    while cursor < TARGET_MS:
        silence_len = random.randint(MIN_SIL_MS, MAX_SIL_MS)
        start = cursor + silence_len
        if start >= TARGET_MS:
            break

        bf, seg = random.choice(bits)
        seg = auto_fix(seg)

        if apply_styles:
            seg = maybe_stretch(seg).fade_in(FADE_MS).fade_out(FADE_MS)
            if random.random() < ECHO_CHANCE:
                seg = apply_echo(seg)
            seg = apply_reverb(seg)
            seg = seg.pan(random.uniform(*PAN_RANGE))
            seg = seg.apply_gain(random.uniform(*GAIN_RANGE_DB))

        tracks[t] = tracks[t].overlay(seg, position=start)
        metadata.append({"track": t, "bit": bf, "start_ms": start, "dur_ms": len(seg)})
        cursor = start + len(seg)

    track_path = os.path.join(OUT_DIR, f"track{t}.wav")
    tracks[t].export(track_path, format="wav")
    print(f"  track {t} written → {track_path}")

# --- master mix ------------------------------------------------------------
print("\nMixing master …")
hum = (
    Sine(HUM_FREQ_HZ)
    .to_audio_segment(duration=TARGET_MS)
    .apply_gain(HUM_GAIN_DB)
    .set_channels(CHANNELS)
)

master = hum
for tr in tracks:
    master = master.overlay(tr)

master_path = os.path.join(OUT_DIR, "master.wav")
master.export(master_path, format="wav")
print(f"✔  master written → {master_path}")

# --- optional CSV ----------------------------------------------------------
if write_csv and metadata:
    csv_path = os.path.join(OUT_DIR, "collage_debug.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metadata[0].keys())
        w.writeheader()
        w.writerows(metadata)
    print(f"✔  CSV saved      → {csv_path}")

print("\nDone – copy the master path above if you need it.")
