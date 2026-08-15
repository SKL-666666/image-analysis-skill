# 🖼️ image-analysis — Structured Image Analysis for Text-Only LLMs

Give **text-only AI models (DeepSeek, GLM, etc.) the ability to "see" images**: OCR text extraction + shape/color/control/icon recognition + table recognition + layout & hierarchy analysis, output as a **structured Markdown report**.

> In one sentence: it *translates* an image into a text report a model can read — so text-only models can understand images too.

Great for: **UI screenshots, web pages, flowcharts, architecture diagrams, tables & reports, form pages, chat transcripts, text documents**. For photos/posters it provides color palette and layout (limited content understanding).

---

## ✨ Features

| Capability | Description |
|---|---|
| 🔤 Dual-engine OCR | Windows built-in OCR + RapidOCR run in parallel, de-duplicated and merged for higher accuracy |
| 🔷 Shape recognition | Rectangles / circles / lines / buttons / badges, with coordinates (%), colors, and guessed functions |
| 🧱 Layout & hierarchy | Region grouping + parent–child containment tree (which element lives in which card) |
| 📊 Table recognition | Grid tables (with vertical lines) + borderless tables (columns inferred from text positions) |
| 🎨 Color character map | Rendered with text color codes so models can read the layout directly |
| 📐 Formula normalization | Math symbols normalized (x², ≥, −, etc.) |
| ⚡ Caching & batching | MD5-based content cache for instant repeat analysis; multi-image/directory parallel processing |
| 🎛️ Zero config | No setup, works out of the box on Windows (`install_deps.bat` installs dependencies) |

## 🧠 How it works

Mainstream LLMs (e.g., DeepSeek) are **text-only** — they cannot take image input. This tool does the image understanding **locally**:

```
image → dual-engine OCR + shape/color/table/icon/layout analysis → 8-section Markdown report → fed to the model
```

The report carries text, coordinates, colors, structure, and hierarchy, so the model can answer "what's in this image?" accurately.

## 📦 Requirements

- **Windows 10/11** (OCR uses the built-in Windows engine)
- **Python 3.9+** (check *Add Python to PATH* during install)
- Chinese language pack (optional but recommended): Settings → Time & Language → Language, include Chinese (otherwise OCR only recognizes English)
- Optional speed-up: `python -m pip install rapidocr_onnxruntime` (~100 MB); dual-engine mode activates automatically once installed

## 🚀 Install (2 steps)

```bat
:: 1. Install Python 3.9+ (see above)
:: 2. Run install_deps.bat — equivalent to:
python -m pip install pillow numpy opencv-python-headless
```

Verify: `python scripts\analyze_image.py any-image.png --plain` — you should see the 8-section report.

## 🤖 Use as an Agent Skill

The `scripts/` folder is a **self-contained skill package** (`SKILL.md` definition + scripts) that can be imported into any client supporting Agent Skills, giving your agent "vision":

```bash
# ① Install dependencies first (either way)
install_deps.bat            # double-click on Windows
# or: python -m pip install pillow numpy opencv-python-headless

# ② Copy scripts/ as a skill (ZCode example)
cp -r scripts ~/.zcode/skills/image-analysis      # ZCode
# DeepSeek Harness (DSH):
cp -r scripts ~/.dsh/skills/image-analysis
# Claude Code / Codex:
cp -r scripts ~/.claude/skills/image-analysis     # or ~/.codex/skills
```

Then send the agent a **local path to an image or screenshot** and ask (e.g., "analyze this image", "what's in this screenshot"). The agent follows the flow in `SKILL.md`, runs the script automatically, and answers from the structured report.

> ⚠️ **Compatibility note**: some agent tools **refuse to send images to single-modal (text-only) models** — the image content is silently dropped or errors out. This skill works around that by doing local analysis and passing a text report. **Currently verified on ZCode only**; please test the behavior in other clients before relying on it.

> 💡 The skill folder name becomes the skill name (rename freely); the `description` field in `SKILL.md` decides when the agent auto-triggers the skill.

## 📖 Quick start

```bash
# Single image (--plain recommended: text color codes are easier for models to read)
python scripts\analyze_image.py screenshot.png --plain

# Batch in parallel (multiple images or a directory, 3 workers by default)
python scripts\analyze_image.py img1.png img2.png some-dir --plain --out-dir results --workers 4

# Dark-theme images / small images are handled automatically, no extra flags
```

### CLI options

| Option | Description | Default |
|---|---|---|
| `image paths...` | One or more images or directories (auto-scans png/jpg/jpeg/webp/bmp) | required |
| `--plain` | Character map as `[#RRGGBB]` text color codes (model-friendly) | off (ANSI codes) |
| `--width N` | Character map width | 72 |
| `--no-ascii` | Skip the character map (saves tokens) | off |
| `--out file` | Output file for a single image | `<image>_analysis.md` |
| `--out-dir dir` | Output directory for batch runs | current directory |
| `--ocr {auto,windows,rapid,both}` | OCR engine selection | auto |
| `--workers N` | Parallel worker processes | 3 |
| `--cache-dir dir` | Report cache directory (MD5-keyed) | ~/.cache/image-analysis |
| `--no-cache` | Disable cache (force re-analysis) | off |

## 📋 Output report (8 sections)

1. **Confidence** — high/medium/low, determines how strongly the data can be cited
2. **Overview** — original dimensions, dominant colors with coverage
3. **Element list** — parent container, guessed function, type/shape, coordinates, size, color, text
4. **Layout structure** — top bar / sidebar / content containers / connections / icons
5. **Hierarchy** — parent–child containment tree
6. **Text block clustering** — text grouped by paragraph/card
7. **Table recognition** — auto-converted to Markdown tables
8. **Full OCR text** — all text with positions; formula lines tagged `[formula]`

## ⚠️ Limitations (stated honestly)

- **Icon recognition**: 15 built-in common icons (search/settings/menu/close/back/plus/QR-camera/share/favorite/like/download/more/play/user/clock/delete); mismatched styles are skipped (better to miss than to guess wrong)
- **Photos / complex illustrations**: only dominant colors and color-block layout; content understanding is limited
- **Small / stylized text**: <10px text is hard for both engines; artistic fonts are unreliable
- **Complex formulas**: common symbols are normalized, but complex formulas still need human or vision-model confirmation
- **Confidence tiers**: the report distinguishes "recognized" from "inferred" (function column is a guess) — cite accordingly

## 📁 Project structure

```
image-analysis/
├── README.md               ← this document
├── install_deps.bat        ← one-click dependency installer (Windows)
└── scripts/
    ├── SKILL.md            ← skill definition (Agent Skills entry; import scripts/ as the skill root)
    ├── analyze_image.py    ← main script (all logic, zero project dependencies)
    └── ocr.ps1             ← Windows built-in OCR engine wrapper
```

> 📦 Ready-to-use zip package (README + skill + scripts + installer, unzip & go) is not stored in the repo — grab it from **GitHub Releases**: [v1.0.0 · image-analysis-skill-v1.0.0.zip](https://github.com/SKL-666666/image-analysis-skill/releases/latest)

## 🤝 Usage tips

- **As a ZCode Agent Skill**: import the `scripts/` directory into your skill folder (see "Use as an Agent Skill" above); `SKILL.md` contains the full workflow
- **Pairs well with multimodal models**: this tool does the structured extraction; a vision model does the semantic understanding

## 📜 License

[MIT](LICENSE)
