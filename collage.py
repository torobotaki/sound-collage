#!/usr/bin/env python3
"""
collage.py  ─ mix random bits into a 30-s ambience.

Options:
    python collage.py            # normal
    python collage.py --enhance80s   # add gentle mastering polish to final mix
"""
import os, sys, csv, random, time
import numpy as np
import librosa, noisereduce as nr
from pydub import AudioSegment, effects
from pydub.generators import Sine

# ─── CONFIG ────────────────────────────────────────────────────────────────
BITS_DIR = "bits"
OUT_DIR = "out"
OUT_WAV = os.path.join(OUT_DIR, "collage.wav")
DEBUG_CSV = os.path.join(OUT_DIR, "collage_debug.csv")

TARGET_MS = 30_000  # 30 s collage
FADE_MS = 100
MIN_GAP_MS = 100
MAX_GAP_MS = 800
MAX_OVERLAP_MS = 800

EXTRA_BIT_PROB = 0.3  # chance to layer extra bits
MAX_EXTRA_BITS = 3

REVERB_DELAYS = [120, 150, 170]  # ms taps
REVERB_DECAY = 0.1
ECHO_CHANCE = 0.3
ECHO_DELAY_MS = 500

STRETCH_PROB = 0.3
STRETCH_RANGE = (0.8, 1.2)  # ±10 %
GAIN_RANGE_DB = (-3, +3)

HUM_FREQ_HZ = 60
HUM_GAIN_DB = -10
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


# ——— helper FX ————————————————————————————————————————————————
def apply_reverb(seg: AudioSegment) -> AudioSegment:
    out = seg
    for i, d in enumerate(REVERB_DELAYS, 1):
        tap = seg - (i * (1 - REVERB_DECAY) * 10)
        out = out.overlay(tap, position=d)
    return out


def apply_echo(seg: AudioSegment) -> AudioSegment:
    echo = seg - 10  # –10 dB
    return seg.overlay(echo, position=ECHO_DELAY_MS)


def maybe_stretch(seg: AudioSegment) -> AudioSegment:
    if random.random() > STRETCH_PROB:
        return seg
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768
    factor = random.uniform(*STRETCH_RANGE)
    y2 = librosa.effects.time_stretch(y=y, rate=factor)
    data = (y2 * 32767).astype(np.int16).tobytes()
    return AudioSegment(data, frame_rate=seg.frame_rate, sample_width=2, channels=1)


def auto_fix(seg: AudioSegment) -> AudioSegment:
    """
    Very lightweight “doctor” for each bit.
    Fast heuristics decide which treatment to apply:

    • Loud broadband hiss  → spectral-noise reduction
    • Dull 80-ish tape     → rumble cut + air boost + low-mid dip
    • Music-heavy section  → gentle 2 : 1 compression (+2 dB make-up)

    Returns an AudioSegment ready for the rest of the FX chain.
    """
    # numpy view of the mono samples
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    sr = seg.frame_rate

    rms_db = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectrum = np.abs(np.fft.rfft(y))
    hi_energy = spectrum[int(len(spectrum) * 0.7) :].mean()
    lo_energy = spectrum[: int(len(spectrum) * 0.1)].mean()

    out = seg

    # 1) Modern hiss / background TV ⇒ denoise
    if rms_db > -30:  # fairly loud noise-floor
        y_dn = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        out = AudioSegment(
            (y_dn * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )

    # 2) 1980s dull mono ⇒ gentle EQ brighten
    elif centroid < 1_000:  # low spectral centroid
        out = out.high_pass_filter(40)  # rumble cut
        out = out.low_pass_filter(16_000).apply_gain(+2)  # air shelf
        out = out.apply_gain(-2)  # broad low-mid dip

    # 3) Music-heavy ⇒ light compression
    if hi_energy > lo_energy * 2:
        out = effects.compress_dynamic_range(
            out, threshold=-18, ratio=2.0, attack=10, release=250
        )
        out = out.apply_gain(+2)  # make-up gain

    return out


# ————————————————————————————————————————————————————————————————


def enhance_80s(mix: AudioSegment) -> AudioSegment:
    """
    40 Hz HPF  →  +2 dB air  →  tiny pseudo-stereo  →  –3 dB gain.
    Runs safely on stereo or mono input.
    """
    hp = mix.high_pass_filter(40)
    airy = hp.low_pass_filter(16_000).apply_gain(+2)

    # start from ONE mono channel
    left = airy.split_to_mono()[0]  # guaranteed mono
    delay = 15  # ms
    right = AudioSegment.silent(delay) + (left - 10)  # delayed –10 dB
    left = left + AudioSegment.silent(delay)  # pad to same length

    stereo = AudioSegment.from_mono_audiosegments(left, right)
    return stereo.apply_gain(-3)


# ——— load bits ————————————————————————————————————————————————
bit_list = [f for f in os.listdir(BITS_DIR) if f.lower().endswith(".wav")]
bits = [(bf, AudioSegment.from_wav(os.path.join(BITS_DIR, bf))) for bf in bit_list]
print(f"Loaded {len(bits)} bits from '{BITS_DIR}'")

# ——— prepare timeline & hum ————————————————————————————————
hum = Sine(HUM_FREQ_HZ).to_audio_segment(duration=TARGET_MS).apply_gain(HUM_GAIN_DB)
timeline = AudioSegment.silent(duration=TARGET_MS)
placements = []
cursor = 0
print("Building collage:")

while cursor < TARGET_MS:
    gap = random.randint(MIN_GAP_MS, MAX_GAP_MS)
    base = min(cursor + gap, TARGET_MS)
    if base >= TARGET_MS:
        break
    n = 1 + (
        random.randint(1, MAX_EXTRA_BITS) if random.random() < EXTRA_BIT_PROB else 0
    )
    for _ in range(n):
        bf, seg = random.choice(bits)
        seg = auto_fix(seg)  # <-- per-bit doctor
        seg = maybe_stretch(seg)
        seg = seg.apply_gain(random.uniform(*GAIN_RANGE_DB))
        seg = seg.fade_in(FADE_MS).fade_out(FADE_MS)
        seg = apply_reverb(seg)
        echo_f = False
        if random.random() < ECHO_CHANCE:
            seg = apply_echo(seg)
            echo_f = True
        pan = random.uniform(-1, 1)
        seg = seg.pan(pan)
        dur = len(seg)
        ov = random.randint(0, MAX_OVERLAP_MS)
        start = max(0, base - ov)
        timeline = timeline.overlay(seg, position=start)
        placements.append(
            {
                "bit": bf,
                "start": start,
                "dur": dur,
                "overlap": ov,
                "pan": round(pan, 2),
                "gain": round(seg.dBFS, 1),
                "echo": echo_f,
            }
        )
        print(f"  • {bf} @{start}ms ov={ov} pan={pan:+.2f}")
    cursor = base

mix = hum.overlay(timeline)

# optional mastering
if "--enhance80s" in sys.argv:
    print("Applying final 80s enhancement…")
    mix = enhance_80s(mix)

# ——— output ————————————————————————————————————————————————
mix.export(OUT_WAV, format="wav")
print(f"Saved collage to {OUT_WAV}")

with open(DEBUG_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=placements[0].keys())
    w.writeheader()
    w.writerows(placements)
print(f"Debug CSV → {DEBUG_CSV}")
