import os, time, random, csv
import numpy as np
import noisereduce as nr
import pyworld as pw
import librosa
import soundfile as sf
from pydub import AudioSegment, silence

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_DIR = "in"
BITS_DIR = "bits"
DEBUG_CSV = "debug.csv"
CLEAR_BITS = True  # wipe bits_dir at start?
MIN_SILENCE_MS = 150
KEEP_SILENCE_MS = 100
MIN_CHUNK_MS = 100
MAX_CHUNK_MS = 600
MAX_SHIFT_ST = 40.0  # absolute cap, but logic below uses smaller ranges
# ──────────────────────────────────────────────────────────────────────────────

if CLEAR_BITS:
    os.makedirs(BITS_DIR, exist_ok=True)
    for f in os.listdir(BITS_DIR):
        if f.lower().endswith(".wav"):
            os.remove(os.path.join(BITS_DIR, f))
else:
    os.makedirs(BITS_DIR, exist_ok=True)


def estimate_pitch(x, sr):
    x64 = x.astype(np.double)
    f0, timeaxis = pw.harvest(x64, sr, frame_period=5.0)
    f0[f0 == 0] = np.nan
    return float(np.nanmedian(f0)) if not np.all(np.isnan(f0)) else None


def world_pitch_shift(x, sr, semitones):
    x64 = x.astype(np.double)
    f0, timeaxis = pw.harvest(x64, sr, frame_period=5.0)
    sp = pw.cheaptrick(x64, f0, timeaxis, sr)
    ap = pw.d4c(x64, f0, timeaxis, sr)
    ratio = 2 ** (semitones / 12)
    f0_shifted = f0 * ratio
    y = pw.synthesize(f0_shifted, sp, ap, sr)
    return y.astype(np.float32)


def calculate_shift(base, bias_female=0.4):
    """
    Subtler shifts:
    - 40% chance to bias female: +1.0 to +3.0 semitones
    - 60% chance to bias male:   -3.0 to -1.0 semitones
    """
    if base is None or base < 50:
        return None, "unknown"
    if random.random() < bias_female:
        st = random.uniform(5.0, 25.0)
        gender = "female"
    else:
        st = random.uniform(-12.0, -1.0)
        gender = "male"
    st = float(np.clip(st, -MAX_SHIFT_ST, MAX_SHIFT_ST))
    return st, gender


def chop(chunk):
    parts, i = [], 0
    while i < len(chunk):
        part = chunk[i : i + MAX_CHUNK_MS]
        if len(part) >= MIN_CHUNK_MS:
            parts.append((part, i))
        i += MAX_CHUNK_MS // 2
    return parts


# Main processing
debug_rows = []
bit_index = 0
files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav"))
start_all = time.time()

for idx, fname in enumerate(files, 1):
    t0 = time.time()
    print(f"\n[{idx}/{len(files)}] {fname}")
    path = os.path.join(INPUT_DIR, fname)

    # 1) Denoise
    y, sr = librosa.load(path, sr=16000)
    y_d = nr.reduce_noise(y=y, sr=sr)

    # 2) Split on silence
    sf.write("_tmp.wav", y_d, sr)
    audio = AudioSegment.from_wav("_tmp.wav").normalize()
    os.remove("_tmp.wav")

    thresh = audio.dBFS - 14
    segments = silence.split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=thresh,
        keep_silence=KEEP_SILENCE_MS,
    )
    print(f"  → {len(segments)} speech segments")

    # 3) Process segments into bits
    for seg_i, seg in enumerate(segments, 1):
        for sub_i, (subchunk, offset) in enumerate(chop(seg), 1):
            arr = np.array(subchunk.get_array_of_samples()).astype(np.float32) / 32768.0

            base_pitch = estimate_pitch(arr, sr)
            detected_gender = (
                "male"
                if base_pitch and base_pitch < 180
                else "female"
                if base_pitch and base_pitch > 220
                else "unknown"
            )

            shift, target_gender = calculate_shift(base_pitch)

            # Attempt shift, fallback smaller if no pitch-detect
            final_arr, final_pitch, used_shift = None, None, 0.0
            tried = []
            if shift is not None:
                for factor in (1.0, 0.75, 0.5):
                    s = shift * factor
                    tried.append(round(s, 2))
                    y_sh = world_pitch_shift(arr, sr, s)
                    p = estimate_pitch(y_sh, sr)
                    if p:
                        final_arr, final_pitch, used_shift = y_sh, p, s
                        break
                if final_arr is None:
                    final_arr, final_pitch = arr, base_pitch
                    target_gender = "unchanged"
                    tried.append(0.0)
            else:
                final_arr, final_pitch = arr, base_pitch
                tried.append(0.0)

            shifted_gender = (
                "male"
                if final_pitch and final_pitch < 180
                else "female"
                if final_pitch and final_pitch > 220
                else "unknown"
            )

            bit_name = f"bit_{bit_index:04d}.wav"
            sf.write(os.path.join(BITS_DIR, bit_name), final_arr, sr)

            debug_rows.append(
                {
                    "bit_file": bit_name,
                    "original_file": fname,
                    "start_ms": offset,
                    "duration_ms": len(subchunk),
                    "base_pitch_Hz": round(base_pitch, 1) if base_pitch else None,
                    "shift_semitones": round(used_shift, 2),
                    "final_pitch_Hz": round(final_pitch, 1) if final_pitch else None,
                    "detected_gender": detected_gender,
                    "target_gender": target_gender,
                    "shifted_gender": shifted_gender,
                    "shift_tried": ",".join(f"{v:.2f}" for v in tried),
                }
            )

            print(
                f"    [{seg_i}.{sub_i}] {bit_name} | {len(subchunk)}ms | "
                f"{base_pitch or 'None'}Hz → {used_shift:+.2f}st → "
                f"{final_pitch or 'None'}Hz (tried {tried})"
            )
            bit_index += 1

    print(f"  → done in {time.time() - t0:.1f}s")

print(f"\n🏁 Completed in {time.time() - start_all:.1f}s — {bit_index} bits total.")

# Write debug CSV
with open(DEBUG_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
    writer.writeheader()
    writer.writerows(debug_rows)

print(f"Debug saved to {DEBUG_CSV}")
