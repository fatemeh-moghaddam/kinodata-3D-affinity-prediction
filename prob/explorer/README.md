# Probe Sweep Explorer

An interactive page for comparing probe runs across every factor of the sweep —
target, GNN, layer, probe model, RMSD threshold and split type.

Output is **one self-contained HTML file**. 
The runs move inside the page as JSON. 


## Building it

```bash
uv run python -m prob.explorer --open
```

Writes `data/probing/probe_explorer.html` by default. Rebuild it whenever new
runs land — it re-indexes the sweep from disk each time.

Narrow it to one slice of the sweep (each flag takes several values):

```bash
uv run python -m prob.explorer --target affinity mw --gnn CGNN-3D DTI -o figs/affinity.html
```

`--gnn --target --probe --split --rmsd --layer` are forwarded to
`find_probe_runs`, so the page contains exactly the runs you would have got from
`find_probe_runs` call. `--root` points at a different probing directory; `--out` sets the
output path; `--open` launches a browser.

### From a notebook:

```python
from prob.explorer import build_explorer, collect_runs

build_explorer("explorer.html", target="affinity")

# or hand it a frame you already have
runs = collect_runs(gnn_model_type="CGNN-3D")
build_explorer("cgnn3d.html", runs=runs[runs.layer > 0])
```

## What the views are for

| View | Question it answers |
|---|---|
| **Depth curves** | Does the property get *built up* through the layers, or was it already in the input features? The main probing figure. Lays out as a panel grid — see below. |
| **Coverage matrix** | How does every condition compare on one metric, and which combinations never ran? Blank cells are shown explicitly. |
| **Ranked intervals** | Which runs are actually distinguishable? Dot + 95% bootstrap CI + the paired shuffled-ident baseline tick. |
| **Table** | The underlying numbers, sortable, with copy-to-CSV. |

## Laying out the depth curves

The curves take a **panel grid**, set by two pickers:

- **Panel rows per** and **Panel columns per** — set both for a 2-D facet grid
  (rows = Target, columns = RMSD), the same anatomy as `plot_run_grid` in
  `prob/plot_multipanel.py`, so a screenshot and the matplotlib figure read the
  same way. Combinations that never ran are drawn as a dashed `no run` cell
  rather than left as a hole.
- Set only one and the panels wrap into a strip; **Panels per row**
  (Auto / 1–4) then chooses the shape. It is disabled with two facets, where the
  column factor already fixes the column count. If the number you ask for would
  squeeze panels below the width a curve stays readable at, it draws fewer and
  says so.
- Picking a factor the other facet already holds **swaps** the two rather than
  putting one factor on both axes.

**Y scale** has three modes, and the choice matters more than it looks:

| Mode | Use it when |
|---|---|
| **Per panel** | Reading the shape of each curve; panels are not comparable. |
| **Per row** | A 2-D grid — each row gets its own range, so columns stay comparable without the widest-ranging row flattening the rest. Offered only when a row factor is set. |
| **Shared** | Comparing absolute heights everywhere; one range for every panel. |

With rows = Target, `Shared` lets `mw` (R² up to 1.0) flatten every other row,
and `Per panel` makes the columns within a row incomparable. `Per row` is
usually what you want for a target × something grid.

## When several runs land on one mark

A factor that is not on the x axis, not the colour, not a facet and not filtered
to a single level still has several levels in view — so several runs share one
point or one cell. Averaging them is usually wrong (an R² for molecular weight
has no business being averaged with one for affinity), so the page refuses to do
it quietly.

It names the leftover factor in a **CHOOSE** banner above the chart, with
one-click ways out: put it on the panel rows, on the panel columns, on the
colour, keep a single level, or average anyway. Putting it on a facet axis moves
whatever was there onto the free axis rather than stranding it.

**Runs on one mark** picks the behaviour:

| Mode | What you get |
|---|---|
| **Split** *(default)* | One line per run — each combination of the leftover factors drawn separately, thinned. Nothing averaged, nothing hidden. |
| **Mean** / **Median** | Collapsed to one line, every collapsed point marked with a red star ★ and `n=`. |
| **Break** | No value placed at all: a red star where runs collide and the line breaks around it. |

In the **coverage matrix** a cell holds one number, so there is nothing to split:
under Split or Break a multi-run cell shows the star and its count instead of a
value, and Mean/Median fill the number in with the star still on it.

The star is red, carries a hover title giving the count, and is named in the
banner text — it never signals by colour alone.

## Getting a figure out

Each chart card has **theme · scale · Save PNG** in its header.

- **Theme** defaults to **Light**, independent of the theme you are browsing in —
  so a light figure for the thesis comes out of a dark page without switching
  anything. `As shown` exports what is on screen. Changing the theme redraws the
  chart, exports, then puts your view back.
- **Scale** defaults to **3×**, which is roughly 300 dpi at the on-screen size.
- **Where it saves**, in order of what the browser supports:
  1. **Chrome / Edge, including on `file://`** — a real save dialog, so you pick
     the folder and the name. This is the path you want for thesis figures.
  2. **Firefox / Safari** — no such API; the file lands in your downloads folder.
  3. **The hosted Artifact copy** — saves through the viewer's own confirmation
     prompt, which you can decline. That sandbox blocks ordinary page downloads,
     so this route is the only one that works there; the page detects which case
     it is in rather than failing quietly.

  Either way the filename says what the figure is, e.g.
  `probe_curves_r2_by-layer_colour-gnn_model_type_panels-target_mlp_2_random-k-fold.png`.

The export inlines every computed style onto a clone of the SVG before
rasterizing — without that, axis and label colours come from the stylesheet and
are lost. Text is exported in a system font stack: remote webfonts do not load
inside a rasterized SVG, so IBM Plex would silently fall back anyway.

## Reading the numbers honestly

- **One run is one point estimate.** There are no per-fold predictions on disk
  (fold is collapsed at aggregation), so a box plot here would have nothing to
  spread. The interval on each mark is the bootstrap CI the pipeline already
  wrote to `reports/<probe>_summary.json` — that is the real uncertainty.
- **R² is the only cross-target metric.** RMSE and MAE carry each target's own
  units. Select one of them across several targets and the page says so.
- **Nothing is averaged behind your back.** See below.
- **Runs missing a factor are set aside**, not silently hidden. A few legacy
  paths predate the `<layer>` directory level; the build prints how many and the
  page footer repeats it.

## Extending it

Everything the page knows lives in `build.py` — the JavaScript reads it from the
payload and hard-codes no column names.

**A new factor** (say `seed`), once it is a column on the runs frame:

```python
FACTORS = [
    ...,
    Factor("seed", "Seed"),
]
```

It appears as a filter, and as an option in every rows/columns/colour-by/facet
picker, with no template change.

**A new metric** — add the column in `collect_runs`, then:

```python
METRICS = [
    ...,
    Metric("pearson", "Pearson r", higher_better=True, cross_target=True),
]
```

Give it `ci_lower=`/`ci_upper=` and the depth curves draw a band and the ranked
view draws a whisker for it too. Give it `baseline=` and the baseline tick and
the "CI clears baseline" stat follow. `higher_better=False` flips the ranking
and the colour ramp; `cross_target=False` turns on the units warning.

**A different opening view** — `DEFAULT_FILTER`, `DEFAULT_METRIC` and
`DEFAULT_VIEW` at the top of `build.py`. `DEFAULT_VIEW` carries the depth-curve
grid (`facetRowBy`, `facetColBy`, `panelsPerRow`, `yScale`) and the matrix axes;
`""` is a valid facet, meaning "unset". A default naming something that is not
present is ignored rather than producing a blank page.

**Layout, colour, copy** — `template.html`. The palette is defined once as CSS
custom properties at the top and both themes are declared there; the categorical
slots are a validated colourblind-safe order, so re-order them rather than
substituting arbitrary hues.

## Tests

```bash
uv run --with pytest --with pytest-cov python -m pytest prob/tests/test_explorer_build.py --no-cov --noconftest
```

`--noconftest` is needed only because `prob/tests/conftest.py` imports the GNN
stack (`torch_scatter`), which the explorer does not use. Drop the flag once
that import resolves in your environment.
