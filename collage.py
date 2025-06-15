import os
import csv
import time
import random
import numpy as np
import librosa
from pydub import AudioSegment, silence
from pydub.generators import Sine

# ─── CONFIG ────────────────────────────────────────────────────────────────
BITS_DIR = "bits"
OUT_DIR = "out"
COLLAGE_WAV = os.path.join(OUT_DIR, "collage.wav")
DEBUG_CSV = os.path.join(OUT_DIR, "collage_debug.csv")
TARGET_DURATION = 100_000  # ms
MIN_GAP_MS = 100  # min before next group
MAX_GAP_MS = 800  # max before next group
MAX_OVERLAP_MS = 800  # how early within gap to start
FADE_MS = 100  # per‐bit fade in/out
REVERB_DELAYS = [50, 100, 150]
REVERB_DECAY = 0.5
ECHO_CHANCE = 0.3
ECHO_DELAY_MS = 200
EXTRA_BIT_PROB = 0.3
MAX_EXTRA_BITS = 3
STRETCH_PROB = 0.2  # 20% chance to stretch
STRETCH_RANGE = (0.6, 1.6)
GAIN_RANGE_DB = (-3.0, +3.0)
HUM_FREQ_HZ = 60
HUM_GAIN_DB = -30
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# load bits
bit_files = sorted(f for f in os.listdir(BITS_DIR) if f.lower().endswith(".wav"))
print(f"Loaded {len(bit_files)} bits from '{BITS_DIR}'")
bits = [(bf, AudioSegment.from_wav(os.path.join(BITS_DIR, bf))) for bf in bit_files]


def apply_reverb(seg: AudioSegment) -> AudioSegment:
    out = seg
    for i, d in enumerate(REVERB_DELAYS, start=1):
        tap = seg - (i * (1 - REVERB_DECAY) * 10)
        out = out.overlay(tap, position=d)
    return out


def apply_echo(seg: AudioSegment) -> AudioSegment:
    echo = seg - 10
    return seg.overlay(echo, position=ECHO_DELAY_MS)


def maybe_stretch(seg: AudioSegment) -> AudioSegment:
    if random.random() > STRETCH_PROB:
        return seg
    # to numpy mono
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    sr = seg.frame_rate
    factor = random.uniform(*STRETCH_RANGE)
    y2 = librosa.effects.time_stretch(y=y, rate=factor)
    # back to AudioSegment
    data = (y2 * 32767).astype(np.int16).tobytes()
    return AudioSegment(data, sample_width=2, frame_rate=sr, channels=1)


# prepare hum background
hum = (
    Sine(HUM_FREQ_HZ).to_audio_segment(duration=TARGET_DURATION).apply_gain(HUM_GAIN_DB)
)

# prepare empty timeline
timeline = AudioSegment.silent(duration=TARGET_DURATION)
placements = []

cursor = 0
print("Building collage:")
while cursor < TARGET_DURATION:
    gap = random.randint(MIN_GAP_MS, MAX_GAP_MS)
    start_base = min(cursor + gap, TARGET_DURATION)
    if start_base >= TARGET_DURATION:
        break

    n_bits = 1
    if random.random() < EXTRA_BIT_PROB:
        n_bits += random.randint(1, MAX_EXTRA_BITS)

    for _ in range(n_bits):
        bf, seg = random.choice(bits)
        dur = len(seg)
        overlap = random.randint(0, MAX_OVERLAP_MS)
        start = max(0, start_base - overlap)

        piece = seg.fade_in(FADE_MS).fade_out(FADE_MS)
        piece = apply_reverb(piece)
        echo_flag = False
        if random.random() < ECHO_CHANCE:
            piece = apply_echo(piece)
            echo_flag = True

        piece = maybe_stretch(piece)

        gain = random.uniform(*GAIN_RANGE_DB)
        piece = piece.apply_gain(gain)

        pan = random.uniform(-1.0, 1.0)
        piece = piece.pan(pan)

        timeline = timeline.overlay(piece, position=start)

        placements.append(
            {
                "bit_file": bf,
                "start_ms": start,
                "duration_ms": dur,
                "overlap_ms": overlap,
                "pan": round(pan, 2),
                "gain_db": round(gain, 1),
                "reverb": True,
                "echo": echo_flag,
                "stretched": round(len(piece) / dur, 2),
            }
        )
        print(
            f"  » Placed {bf} @ {start}ms (dur={dur}ms, ov={overlap}ms, pan={pan:.2f}, gain={gain:+.1f}dB, stretch×{placements[-1]['stretched']}, echo={echo_flag})"
        )

    cursor = start_base

print(f"Done: {len(placements)} placements over {TARGET_DURATION}ms")

# mix with hum
final_mix = hum.overlay(timeline)

# export
print(f"Exporting to '{COLLAGE_WAV}'…")
final_mix.export(COLLAGE_WAV, format="wav")
print("Export done.")

# write debug CSV
with open(DEBUG_CSV, "w", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=list(placements[0].keys()))
    writer.writeheader()
    writer.writerows(placements)
print(f"Debug written to '{DEBUG_CSV}'")
