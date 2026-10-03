"""Build self-hosted portfolio graphics from the preserved experiment records."""
from pathlib import Path
import html
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NAV = ROOT / "projects/tactile-navigation"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
INK, PANEL, WHITE, MUTED, TEAL, GOLD = "#101f2a", "#172c39", "#f4f0e7", "#a6b8c3", "#65d0c2", "#d9b879"
data = json.loads((NAV / "outputs/sky_results.json").read_text())
rows = data["evaluation_records"]


def text(x, y, value, size=18, color=WHITE, weight=400, extra=""):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" {extra}>{html.escape(value)}</text>'


def save(name, width, height, items, label):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(label)}</title>' + "".join(items) + "</svg>"
    ET.fromstring(svg)
    (ASSETS / name).write_text(svg, encoding="utf-8")


def hero():
    row = next(r for r in rows if r["room"] == "sky_staggered_baffles")
    record = json.loads((NAV / row["record"]).read_text())
    manifest = json.loads((NAV / "maps/sky_staggered_baffles.json").read_text())
    assert manifest["start_heading_deg"] == 90
    sx, sy = manifest["start_xy"]
    def point(x, y):
        return (765 + x * 60, 334 - y * 60)
    path = []
    for sample in record["samples"][::20] + record["samples"][-1:]:
        if sample["truth"]:
            x, y = point(sample["truth"][0] + sx, sample["truth"][1] + sy)
            path.append(f"{x:.2f},{y:.2f}")
    items = [f'<rect width="1200" height="430" rx="18" fill="{INK}"/>',
             f'<path d="M52 52H100" stroke="{GOLD}" stroke-width="3"/>',
             text(118, 58, "SKY LAN  /  RESEARCH PORTFOLIO", 15, GOLD, 700, 'letter-spacing="2"'),
             text(52, 150, "Walking toward", 59, WHITE, 700, 'letter-spacing="-2"'),
             text(52, 218, "understanding.", 59, WHITE, 700, 'letter-spacing="-2"'),
             text(56, 269, "Exploring how robots move, sense, and adapt.", 20, MUTED),
             f'<line x1="56" y1="312" x2="629" y2="312" stroke="#304653"/>',
             text(56, 346, "MECHANICAL ENGINEERING", 13, GOLD, 700, 'letter-spacing="1.7"'),
             text(56, 374, "Humanoid locomotion  /  Tactile navigation", 17, WHITE),
             f'<rect x="726" y="44" width="432" height="342" rx="14" fill="{PANEL}"/>',
             text(750, 73, "FROM A REAL SIMULATION RECORD", 11, MUTED, 700, 'letter-spacing="1.2"')]
    for x in range(7):
        px, _ = point(x, 0)
        items.append(f'<line x1="{px}" y1="94" x2="{px}" y2="334" stroke="#26404e" stroke-width="1"/>')
    for y in range(5):
        _, py = point(0, y)
        items.append(f'<line x1="765" y1="{py}" x2="1125" y2="{py}" stroke="#26404e" stroke-width="1"/>')
    items.append('<rect x="765" y="94" width="360" height="240" fill="none" stroke="#708897" stroke-width="4"/>')
    for obstacle in manifest["obstacles"]:
        (x1, y1), (x2, y2) = [point(*p) for p in obstacle["pts"]]
        items.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#708897" stroke-width="5"/>')
    items.append(f'<polyline points="{" ".join(path)}" fill="none" stroke="{TEAL}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    x, y = point(sx, sy)
    gx, gy = point(*manifest["goal_xy"])
    items += [f'<circle cx="{x}" cy="{y}" r="5" fill="{TEAL}"/>',
              f'<circle cx="{gx}" cy="{gy}" r="18" stroke="{GOLD}" stroke-width="1.5" fill="none"/>',
              f'<circle cx="{gx}" cy="{gy}" r="4" fill="{GOLD}"/>',
              text(751, 367, "Staggered baffles  ·  executed path  ·  seed 0", 13, MUTED)]
    save("research-cover.svg", 1200, 430, items, "Sky Lan robotics portfolio. A recorded staggered-baffle simulation trajectory appears beside the title Walking toward understanding.")


def summary():
    values = [("03", "NEW MAP DESIGNS", "Distinct obstacle layouts"),
              ("04", "COMPLETED TRIALS", "One original + three new maps"),
              ("04", "PHYSICAL ARRIVALS", "No recorded falls"),
              ("01", "FORMAL SUCCESS", "Arrival belief reached 95%")]
    assert len(rows) == 4 and sum(r["goal_reached"] for r in rows) == 4 and sum(r["success"] for r in rows) == 1
    items = [f'<rect width="1200" height="194" rx="14" fill="{INK}"/>']
    for i, (number, label, detail) in enumerate(values):
        x = 34 + i * 300
        if i:
            items.append(f'<line x1="{i*300}" y1="28" x2="{i*300}" y2="139" stroke="#304653"/>')
        items += [text(x, 77, number, 51, TEAL if i < 3 else GOLD, 700),
                  text(x, 108, label, 13, WHITE, 700, 'letter-spacing="1"'),
                  text(x, 134, detail, 14, MUTED)]
    items.append(text(34, 174, "PILOT SCOPE  /  Seed 0 per map · physical simulation · counts describe these trials only", 13, MUTED))
    save("pilot-at-a-glance.svg", 1200, 194, items, "Pilot study: three new maps, four completed simulations, four physical arrivals, one formal success. One seed per map.")


def section_headers():
    labels = [("study", "01 / THE STUDY", "Can a robot know when it has arrived?"),
              ("direction", "02 / THE MOTIVATION", "Curiosity, motion, and a reason to continue."),
              ("technical", "03 / THE RESEARCH", "The details behind the demonstration."),
              ("videos", "04 / THE EVIDENCE", "Watch the experiments unfold.")]
    for name, label, title in labels:
        items = [f'<rect width="1200" height="160" rx="14" fill="{INK}"/>',
                 f'<path d="M40 31H70" stroke="{TEAL}" stroke-width="3"/>',
                 text(86, 36, label, 13, GOLD, 700, 'letter-spacing="1.8"'),
                 text(40, 104, title, 34, WHITE, 700)]
        save(f"section-{name}.svg", 1200, 160, items, title)


def arrival_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    names = [("room2", "Original room"), ("sky_island_choice", "Central island"),
             ("sky_offset_gate", "Offset opening"), ("sky_staggered_baffles", "Staggered baffles")]
    selected = [next(r for r in rows if r["room"] == name) for name, _ in names]
    fig, ax = plt.subplots(figsize=(12, 4.25), facecolor=INK)
    fig.subplots_adjust(left=.18, right=.805, top=.71, bottom=.19)
    ax.set_facecolor(INK)
    values = [100 * r["final_arrival_probability"] for r in selected]
    colors = [TEAL if r["success"] else GOLD for r in selected]
    ax.barh(range(4), [100]*4, color=PANEL, height=.48)
    ax.barh(range(4), values, color=colors, height=.48)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_yticks(range(4), [n for _, n in names], color=WHITE, fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.tick_params(axis="both", length=0, colors=MUTED, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(95, color=WHITE, linestyle=(0, (3, 3)), linewidth=1.4)
    for i, (v, r) in enumerate(zip(values, selected)):
        ax.text(v - 2, i, f"{v:.2f}%", ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(103, i, "COMPLETE" if r["success"] else "UNCERTAIN", va="center", color=colors[i], fontsize=9, fontweight="bold", clip_on=False)
    fig.text(.035, .905, "FOUR ARRIVALS. ONE CONFIDENT FINISH.", color=WHITE, fontsize=18, fontweight="bold")
    fig.text(.035, .824, "Final belief mass within both the goal region and heading bound", color=MUTED, fontsize=11)
    fig.text(.786, .78, "95% required", color=WHITE, fontsize=10, ha="right")
    fig.text(.035, .057, "Saved runtime probabilities · seed 0 · a belief threshold, not a statistical confidence interval", color=MUTED, fontsize=10)
    fig.savefig(ASSETS / "arrival-confidence.svg", facecolor=INK)
    svg_path = ASSETS / "arrival-confidence.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(ASSETS / "arrival-confidence.png", facecolor=INK, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    hero()
    summary()
    section_headers()
    arrival_chart()
    (ASSETS / "SOURCES.md").write_text("""# Visual sources

All presentation assets are stored inside this private repository. No third-party badge, tracking image, font service or image host is used.

- `research-cover.svg`: simplified display of simulator-truth positions from the staggered-baffle seed-0 canonical evaluation record; map boundaries and dividers come from its authored manifest. Points are sampled for display; the raw record is unchanged.
- `pilot-at-a-glance.svg`: counts from the four completed canonical evaluation records. The interrupted attempt is excluded and remains preserved separately.
- `arrival-confidence.svg` / `.png`: final runtime particle-filter arrival probabilities from `projects/tactile-navigation/outputs/sky_results.json`; the common 95% criterion is shown on a zero-based axis.
- `section-*.svg`: typography and layout only.
- Robot thumbnails in the README and video gallery are existing frames/contact sheets from the executed simulations, linked without alteration.

Rebuild with Python and Matplotlib: `python presentation/build_assets.py`.
""", encoding="utf-8")
    print("Built cover, pilot summary, four section headers, and source-backed arrival chart.")
