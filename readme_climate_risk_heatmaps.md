# Climate Risk Heatmaps — README

A simple, portable setup to compute climate **transition** and **physical** risk heatmaps and explore them in a Streamlit app. This project avoids hard-coded file paths, so anyone can run it on their own machine.

---
## 0) TL;DR (Quick start)

1. **Clone / copy** the whole `Summer_intern/` folder.
2. Confirm the two input Excel files are in the project root:
   - `NGFS_GCAM_Carbon_Emissions_Sectors.xlsx`
   - `NGFS_GCAM_Price_Carbons.xlsx`
3. Open a terminal in `Summer_intern/` and (optionally) create an env:
   ```bash
   # Optional but recommended
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install pandas numpy matplotlib seaborn plotly streamlit xlsxwriter openpyxl
   ```
4. Run the notebooks to (re)generate outputs (see §3). Most people can skip straight to the app if outputs already exist.
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open the URL Streamlit prints (usually `http://localhost:8501`).

---
## 1) What’s in here (big picture)

- **Notebooks (analysis & exports)**
  - Transition risk: computes **Path misalignment (PM)**, **Budget overshoot (BO)**, **Abatement share (AS)**, and **Eq9** (normalized cost) variants.
  - Physical risk (short-term): exports “peak 2023–2030” heatmaps for storylines **HDW** and **SF**, for scenarios **DAPS** and **DIRE**, both **Regions×Sectors** and **10 Categories**, plus **Average (HDW & SF)**.
  - There’s also a **collector notebook** that runs all *short-term physical* exports at once to save time.

- **Streamlit app (`app.py`)**
  - Purely reads files from `outputs/` and renders them. You don’t need to touch the app if the outputs are in place.

- **Portable paths**
  - Everything uses a tiny helper (`lib/paths.py`) to read inputs and write outputs relative to the project folder—no user-specific paths.

---
## 2) Project layout (key files & folders)

```
Summer_intern/
├─ app.py                         # Streamlit app (reads from outputs/)
├─ config.yaml                    # Lists default input file names
├─ lib/
│   └─ paths.py                   # in_data()/in_out()/CFG helpers
├─ notebooks/
│   ├─ Streamlit_TR_PR.ipynb      # Exports for Transition + Physical (streamlit-ready)
│   ├─ NGFS_Transition_Risk_Heatmaps.ipynb  # Transition analysis/exports
│   ├─ NGFS_Physical_Risks_Heatmaps.ipynb   # Physical analysis/exports
│   └─ (collector)_Physical_ShortTerm_All.ipynb  # One-run physical exports (optional)
├─ NGFS_GCAM_Carbon_Emissions_Sectors.xlsx   # INPUT (emissions, by scenario)
├─ NGFS_GCAM_Price_Carbons.xlsx              # INPUT (prices, by scenario)
└─ outputs/                     # CREATED by code; read by the app
   ├─ transition/
   │   ├─ 2020-2030/
   │   │   ├─ pm/ | bo/ | as/
   │   │   │   ├─ data/heatmap_data.csv, heatmap_data_long.csv
   │   │   │   └─ png/heatmap.png
   │   │   └─ eq9_regional/ | eq9_bau20/
   │   │       ├─ r2/ or r0/
   │   │       │   ├─ data/...
   │   │       │   └─ png/heatmap.png
   │   └─ 2020-2050/ (same structure)
   └─ physical_shortterm_DAPS/ and physical_shortterm_DIRE/
       ├─ png/, pdf/, data/ (Regions×Sectors, HDW/SF, Average)
       ├─ png_10cats/, pdf_10cats/, data_10cats/ (10 Categories)
       └─ files like: DAPS__HDW__10cats__peak.png, AVG_HDW_SF_10cats.png, etc.
```

> If `outputs/` already contains these folders and files, you can launch the app right away.

---
## 3) Inputs & config (no hard-coded paths)

**`config.yaml`** holds the input file names. Example:
```yaml
default_inputs:
  emissions_xlsx: NGFS_GCAM_Carbon_Emissions_Sectors.xlsx
  prices_xlsx:    NGFS_GCAM_Price_Carbons.xlsx
```

**`lib/paths.py`** exposes these helpers:
- `CFG` → the parsed YAML config loaded as a dict.
- `in_data(name_or_relpath)` → returns a full path to an input file (relative to the project folder).
- `in_out(relpath)` → returns a full path under `outputs/` (creates parent folders as needed).
- `ensure_out_dirs()` → ensures the standard `outputs/` tree exists.

**Why this matters**: no one needs to edit absolute paths like `C:\Users\...` or `/Users/...`.

---
## 4) What the metrics mean (short and sweet)

**Transition**
- **PM – Path misalignment (A)**: sum of log differences between BAU and NZ emissions.
- **BO – Budget overshoot (Ω)**: amount by which cumulative NZ emissions exceed the (NZ) budget.
- **AS – Abatement share**: share of abatement vs BAU (visuals cap extreme values for readability).
- **Eq9 (normalized cost)**: relative cost to move from BAU→NZ.
  - *Eq9 (regional BAU)*: denominator uses each region’s fixed BAU carbon price.
  - *Eq9 (BAU=$20)*: denominator uses a constant $20/tCO₂.
  - Discount rates: **r2 = 2%**, **r0 = 0%**.

**Physical (short-term, peak 2023–2030)**
- Storylines: **HDW** (Heatwave + Drought + Wildfire), **SF** (Storm + Flood).
- Scenarios: **DAPS**, **DIRE**.
- Layouts: **Regions×Sectors** and **10 Categories**.
- Also: **Average (HDW & SF)** pages per scenario.
- Higher = worse.

---
## 5) Running the notebooks (to generate fresh outputs)

Open the notebook you need in `notebooks/` and run all cells. They already include a small **bootstrap** at the top that finds the project folder and imports `in_data()` / `in_out()`.

- **Transition (PM/BO/AS/Eq9)** → re-generates under:
  - `outputs/transition/2020-2030/...`
  - `outputs/transition/2020-2050/...`

- **Physical (short-term)** → re-generates under:
  - `outputs/physical_shortterm_DAPS/...`
  - `outputs/physical_shortterm_DIRE/...`

- **Collector notebook (optional)**
  - Runs all *short-term physical* exports in one go (both scenarios, both storylines, both layouts, plus averages). Use this when you want to refresh everything at once.

> If a cell complains about a missing file: check `config.yaml` and that the Excel files sit in the project root.

---
## 6) Running the Streamlit app

From the project folder:
```bash
cd Summer_intern
streamlit run app.py
```
Then open the URL shown in the terminal.

**Sidebar controls** let you:
- Pick **Transition** or **Physical**.
- Choose **horizon / metric** (Transition) or **scenario / layout / metric** (Physical).
- Toggle an **interactive heatmap** and change color options.
- Download **PNG / CSV (wide & long) / Excel / ZIP**.
- See **Hot spots** tables and compute **Δ (A − B)** comparisons.
- Click **Refresh data cache** if you updated files on disk.

### Share via ngrok (optional)
Run the app on a fixed port, then tunnel it:
```bash
# Terminal 1
streamlit run app.py --server.port 8501

# Terminal 2
ngrok http 8501
```
Share the `Forwarding` URL printed by ngrok. Keep both terminals open.

---
## 7) How things link together

1. **Notebooks** read inputs using `in_data(...)` and export standardized files using `in_out(...)`.
2. Exports go into **predictable folders** under `outputs/` (see structure in §2).
3. **`app.py`** only **reads** those files. It doesn’t do analysis; it just displays what’s already exported.
4. Because paths are relative to the project folder, **any colleague** can copy the folder and run everything with no path edits.

---
## 8) Common issues & fixes

- **“Heatmap image not found.”**
  - That specific PNG/CSV hasn’t been generated. Run the matching notebook cell for that horizon/metric/scenario/layout.

- **“ModuleNotFoundError: lib.paths”**
  - You ran a middle cell without the bootstrap. Re-run the notebook from the top so it can find `lib/` and `config.yaml`.

- **Permission or port in use** when launching Streamlit
  - Try another port: `streamlit run app.py --server.port 8502`.

- **New machine, missing packages**
  - Reinstall deps (see §0) or use a fresh virtual environment.

---
## 9) Hand-off checklist (for colleagues)

- ✅ Copy the **entire** `Summer_intern/` folder.
- ✅ Make sure the two Excel inputs are in the project root.
- ✅ (Optional) Create a Python virtual env and `pip install` the listed packages.
- ✅ Run `streamlit run app.py`. If a view is empty, run the corresponding notebook to create its outputs.

---
## 10) Updating data / adding scenarios

- Put new input files in the project root and edit `config.yaml`:
  ```yaml
  default_inputs:
    emissions_xlsx: NGFS_GCAM_Carbon_Emissions_Sectors.xlsx
    prices_xlsx:    NGFS_GCAM_Price_Carbons.xlsx
  ```
- Re-run the notebooks. The app will pick up the new exports.

---
## 11) Minimal technical notes (for the curious)

- The app does a small bootstrap to locate the project folder (finds `lib/` + `config.yaml`).
- PM/BO/AS heatmaps use per-panel **p97** color cap; Eq9 uses **r2** and **r0** variants.
- Physical short-term outputs include two naming styles for 10-cats PNG/PDF; the app supports both.

---
## 12) Support

Ping the team if something breaks. When reporting an issue, please share:
- The selection you were viewing (e.g., `Transition → PM → 2020-2030`).
- A screenshot of the app and (if relevant) any notebook traceback.
- Your OS and Python version (`python --version`).

