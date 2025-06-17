## README.md

# Sound-collage Toolkit  
_Python utilities to slice, clean, pitch‑shift & re‑compose sound files. Its intended use is with speech recordings, but it can be used with any sound source in theory._

---

###1·Overview
* **`preprocess.py`**  
• Reads raw `.wav` files from **`in/`**  
• Mono‑downs, resamples to 16kHz, runs light noise‑reduction  
• Splits on silence, keeps only chunks300–2000ms  
(≤1000ms→100ms fades,>1000ms→300ms fades)  
• Safe±4st pitch‑shift (PyWorld)  
• Exports numbered bits to **`bits/`** and logs to `debug.csv`.

* **`collage.py`**  
• Pulls bits from **`bits/`**  
• Lays them on beat‑grid tracks (default10,30s) driven by a *pattern string*  
• Optional FX (`--apply-styles`) and custom BPM (`--bpm`,default120)  
• Writes `track0.wav` … `track<N-1>.wav` and `master.wav`
inside a timestamped **`out/`** folder.

---
###2·QuickStart
```bash
# clone & create virtual‑env
git clone https://github.com/yourname/terirem-toolkit.git
cd terirem-toolkit
python3 -m venv venv && source venv/bin/activate

# install Python deps
pip install -r requirements.txt

# add some WAVs
mkdir in bits   # drop .wav files into ./in/

python preprocess.py          # → snippets in ./bits/
python collage.py             # → mix in ./out/

# with FX & custom pattern/BPM
python collage.py --apply-styles --pattern "SSLCCMMhhDD" --bpm 100
```
---
###3·Command‑Line Flags

####`preprocess.py`
| flag | description |
|------|-------------|
| _(none)_ | process every `.wav` in **`in/`** → `bits/` |

####`collage.py`
| flag | default | description |
|------|---------|-------------|
| `--pattern STR`  | `"M"` | Section pattern: `L``M``H``S``C``D` (upper‑case=all tracks), `l``m``h` (solo track‑0) |
| `--bpm N`        | `120` | Beat grid (ms per beat = 60000/BPM) |
| `--apply-styles` | off   | Enable reverb, echo, pan, stretch, ±gain |
| `--csv`          | off   | Dump placement log `collage_debug.csv` |
| `--help`         | –     | Show extended pattern syntax |

---
###4·InstallingDependencies
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install ffmpeg libsndfile1 build-essential

# macOS (Homebrew)
brew install ffmpeg libsndfile

# Python packages
pip install -r requirements.txt
```
*`ffmpeg` is required by PyDub; `libsndfile` speeds up `soundfile`.*

---
###5·RepoLayout
```
terirem-toolkit/
├─ in/            # raw source WAVs
├─ bits/          # snippets (created by preprocess)
├─ out/
│   └─ collage_YY.MM.DD_HH.MM_[s|ns]/   # each mix run
├─ preprocess.py
├─ collage.py
├─ requirements.txt
└─ README.md
```

---
###6·Troubleshooting
| issue | fix |
|-------|-----|
| silence / white noise in bits | ensureFFmpeg inPATH and`libsndfile` installed |
| “PyWorld not found” | `pip install pyworld==0.3.4` |
| master clipping | reduce track count or drop `--apply-styles`, then normalise externally |

---
###7·License
MIT License ©2025 Dialekti Valsamou-Stanislawski
