## README.md

# Sound-collage Toolkit  
_Python utilities to slice, clean, pitch‑shift & re‑compose sound files. Its intended use is with speech recordings, but it can be used with any sound source in theory._

---

## 1. Overview

### preprocess.py

* Reads raw `.wav` files from **`in/`**
* Mono‑down‑mixes, resamples to 16 kHz, applies light noise‑reduction
* Splits on silence, keeps only chunks **300 – 2000 ms**

  * ≤ 1000 ms → **100 ms** fades ‑‑> body ‑‑> **100 ms** fades
  * \>  1000 ms → **300 ms** fades
* Safe ± 4 st pitch‑shift (PyWorld) into male / female range
* Exports numbered bits to **`bits/`** and logs to `debug.csv`

### collage.py

* Pulls bits from **`bits/`**
* Lays them on a beat‑grid (default **10 tracks**, **30 s**) driven by a *pattern string*
* Optional FX (`--apply-styles`) and custom tempo (`--bpm`, default **120 BPM**)
* Writes `track0.wav` … `track<N‑1>.wav` and `master.wav` inside a timestamped **`out/`** folder

---

## 2. Installing

### System packages

| OS                   | Command                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| **Ubuntu / Debian**  | `sudo apt update && sudo apt install ffmpeg libsndfile1 build-essential` |
| **macOS (Homebrew)** | `brew install ffmpeg libsndfile`                                         |

`ffmpeg` is required by **PyDub** – it actually reads/writes the audio.
`soundfile` will use `libsndfile` for a very fast C backend.

### Python packages

Create a fresh v‑env, then:

```bash
pip install -r requirements.txt
```

`pyworld` is optional; if the wheel fails on your platform the scripts still run (pitch‑shift skipped).

---

## 3. Quick Start

```bash
# clone & set up
git clone https://github.com/torobotaki/sound-collage.git
cd terirem-toolkit
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# folders expected by the scripts
mkdir in bits           # drop .wav files into ./in/

# step 1 – slice input
python preprocess.py

# step 2 – build a collage (default 30 s, no FX)
python collage.py

# with FX + custom pattern at 100 BPM
python collage.py \
    --apply-styles \
    --pattern "SCMCHhMDDS" \
    --bpm 20
```

The mix appears in `./out/collage_YY.MM.DD_HH.MM_[s|ns]/`.

---

## 4. Pattern Reference (collage.py)

The total duration is sliced into equal‑length **sections** (one per pattern symbol).

| Symbol | Meaning            | Spawn‑probability | Gain curve         | Active tracks    |
| ------ | ------------------ | ----------------- | ------------------ | ---------------- |
| `S`    | Silence / ambience | **0.01**          | fixed −12 dB       | all              |
| `L`    | Low energy         | **0.3**           | keep previous      | all              |
| `M`    | Medium energy      | **0.6**           | “”                 | all              |
| `H`    | High energy        | **0.9**           | “”                 | all              |
| `C`    | Crescendo          | previous          | ramp up to +6 dB   | all              |
| `D`    | Diminuendo         | previous          | ramp down to −6 dB | all              |
| `l`    | Solo‑low           | **0.3**           | keep previous      | only **track 0** |
| `m`    | Solo‑medium        | **0.6**           | “”                 | only **track 0** |
| `h`    | Solo‑high          | **0.9**           | “”                 | only **track 0** |

*Upper‑case → affects every track; lower‑case → solo track 0.*

> **Beat‑grid**: beat length = `60 000 / BPM` ms (120 BPM → 500 ms). Spawn‑probability is evaluated **per beat**; overlaps are allowed, so higher energy means denser texture.

Example: `--pattern "SSLCCMChhDDLS"` with default 30s creates 12 equal sections of 2.5s each.

---

## 5. Command‑Line Flags

### preprocess.py

| flag     | description                                                        |
| -------- | ------------------------------------------------------------------ |
| *(none)* | process every `.wav` in **`in/`** and export chunks to **`bits/`** |

### collage.py

| flag             | default | description                                    |
| ---------------- | ------- | ---------------------------------------------- |
| `--pattern STR`  | `"M"`   | Section pattern string (see table above)       |
| `--bpm N`        | `120`   | Tempo (ms per beat = 60 000 / BPM)             |
| `--apply-styles` | *off*   | Enable reverb, echo, pan, stretch, random gain |
| `--csv`          | *off*   | Dump placement log `collage_debug.csv`         |
| `--help`         | –       | Show extended pattern syntax                   |

---

## 6. Repo Layout

```
terirem-toolkit/
├─ in/            # raw source WAVs (input)
├─ bits/          # snippets (output of preprocess)
├─ out/
│   └─ collage_YY.MM.DD_HH.MM_[s|ns]/  # each mix run
├─ preprocess.py
├─ collage.py
├─ requirements.txt
└─ README.md
```

---

## 7. Troubleshooting

| Issue                         | Fix                                                                   |
| ----------------------------- | --------------------------------------------------------------------- |
| Silence / white noise in bits | ensure **FFmpeg** is on PATH and `libsndfile` installed               |
| “PyWorld not found”           | `pip install pyworld==0.3.4`                                          |
| Master clipping               | lower track count or drop `--apply-styles`, then normalise externally |

---

## 8. License

MIT License © 2025 Dialekti Valsamou-Stanislawski
