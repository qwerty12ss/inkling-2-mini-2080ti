"""Render the published figures from the CSV files shipped alongside this script."""

import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data"
DEST = ROOT / "assets" / "plots"
BG = "#FAF9F5"
INK = "#162F37"
MUTED = "#4D6267"
GRID = "#D9DFDA"
TEAL = "#00756A"
RUST = "#A94B28"
PALE = "#C4D7D1"
GOLD = "#D48540"
LENGTHS = (512, 32768, 131072, 524032)
LABELS = ("512", "32K", "128K", "512K")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "axes.titlecolor": INK, "axes.edgecolor": GRID, "axes.axisbelow": True,
})


def read_csv(name):
    with (DATA / name).open(encoding="utf-8") as src:
        return list(csv.DictReader(src))


def label(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def canvas(title, caption):
    fig = plt.figure(figsize=(10.5, 5.45))
    fig.text(.065, .913, title, size=24, weight="bold", ha="left", va="top")
    fig.text(.065, .82, caption, size=11.5, color=MUTED)
    return fig


def clear(ax):
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=8)


def legends(fig, x, y):
    handles = [Line2D([], [], color=RUST, marker="o", mfc=BG, lw=1.7, label="Без драфтера"),
               Line2D([], [], color=TEAL, marker="o", lw=2.4, label="С DSpark K5")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncols=2, frameon=False,
               columnspacing=1.8, handlelength=1.8, handletextpad=.55, borderaxespad=0)


def save(fig, filename):
    fig.savefig(DEST / filename, dpi=170, facecolor=BG, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def medians(rows, field, mode):
    return [median(float(row[field]) for row in rows
                   if int(row["input_tokens"]) == length and row["mode"] == mode)
            for length in LENGTHS]


def decode(rows):
    fig = canvas("Генерация с оригинальным DSpark K5",
                 "Inkling 2 Mini  ·  2 × RTX 2080 Ti 22 GiB  ·  TP2  ·  3 прохода  ·  T = 1, reasoning high")
    legends(fig, .065, .765)
    ax = fig.add_axes([.12, .215, .71, .445])
    clear(ax)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_xlim(0, 210)
    ax.set_ylim(-.6, 3.7)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([3, 2, 1, 0], LABELS, weight="bold")
    ax.set_xlabel("Выходных токенов в секунду", labelpad=9)
    ax.text(1.05, 1.10, "Ускорение", transform=ax.transAxes, color=MUTED, fontsize=10)
    off, on = (medians(rows, "matched_decode_tok_s", mode) for mode in ("target", "K5"))
    for y, base, draft in zip((3, 2, 1, 0), off, on):
        ax.plot([base, draft], [y, y], color=PALE, lw=4.5, solid_capstyle="round")
        ax.scatter(base, y, s=65, facecolor=BG, edgecolor=RUST, linewidth=2, zorder=3)
        ax.scatter(draft, y, s=68, facecolor=TEAL, edgecolor=TEAL, zorder=3)
        for value, color in ((base, RUST), (draft, TEAL)):
            ax.annotate(label(value, 1), (value, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=color, size=10, weight="bold")
        ax.text(1.05, y, label(draft / base, 2) + "×", transform=ax.get_yaxis_transform(),
                va="center", size=14, weight="bold")
    fig.text(.065, .052, "512 и 32K — медианы трёх пар. 128K и 512K — по одной паре. "
             "На 512K вход содержит 524 032 токена.", fontsize=9, color=MUTED)
    save(fig, "decode.png")


def prefill(rows):
    fig = canvas("Холодный префил по длине входа",
                 "Модель в видеопамяти. Попадания в KV-кэш — 0 во всех запросах.")
    legends(fig, .065, .764)
    for left, field, title in [(.09, "prefill_tok_s", "Входных токенов в секунду"),
                               (.555, "ttft_s", "Время до первого токена, с")]:
        ax = fig.add_axes([left, .215, .38, .385])
        clear(ax)
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_xscale("log", base=2)
        ax.set_xlim(300, 850000)
        ax.set_xticks(LENGTHS, LABELS)
        ax.set_title(title, loc="left", size=13, weight="bold", pad=15)
        for mode, color, line, marker, size in (("target", RUST, "--", "s", 6.5), ("K5", TEAL, "-", "o", 4.2)):
            ax.plot(LENGTHS, medians(rows, field, mode), ls=line, color=color, lw=2,
                    marker=marker, ms=size, mfc=BG if mode == "target" else color)
        if field == "prefill_tok_s":
            ax.set_ylim(0, 5450)
            ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
        else:
            ax.set_yscale("log")
            ax.set_ylim(.09, 465)
            ax.set_yticks([.1, 1, 10, 100, 300])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: label(x, 1 if 0 < x < 1 else 0)))
        ax.minorticks_off()
    fig.text(.065, .064, "Длина входа на логарифмической шкале. Эффективная скорость префила = длина входа / TTFT.",
             size=9.5, color=MUTED)
    save(fig, "prefill.png")


def prefix_cache(rows):
    fig = canvas("Повтор запроса с изображением на 448K",
                 "Время до первого токена по счётчикам движка и число токенов, обработанных заново.")
    ax = fig.add_axes([.258, .24, .49, .44])
    clear(ax)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_xlabel("До первого токена, с", labelpad=8)
    ax.set_xlim(0, 217)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(-.6, 3.6)
    names = {"cold-json": "Холодный запрос", "repeat-json": "Точный повтор",
             "repeat-stream": "Повтор в стриме", "changed-suffix-json": "Изменён конец запроса"}
    ax.set_yticks([3, 2, 1, 0], [names[r["case"]] for r in rows])
    for y, row in zip((3, 2, 1, 0), rows):
        time_first = float(row["engine_ttft_s"])
        color = RUST if int(row["cached_tokens"]) == 0 else TEAL
        ax.barh(y, time_first, .36, color=color)
        ax.text(time_first + 2.7, y, label(time_first, 2), va="center", fontsize=10,
                weight="bold", color=color)
        ax.text(1.18, y, label(int(row["computed_tokens"])), transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=14, weight="bold")
    ax.text(1.18, 1.095, "Вычислено заново", transform=ax.transAxes, ha="center", fontsize=10, color=MUTED)
    fig.text(.065, .077, "На точном повторе 456 704 из 458 498 входных токенов взяты из KV-кэша. "
             "Каждый ответ сгенерирован заново.", fontsize=9.3, color=MUTED)
    save(fig, "vision-cache.png")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rows = read_csv("text-pairs.csv")
    decode(rows)
    prefill(rows)
    prefix_cache(read_csv("vision-prefix.csv"))


if __name__ == "__main__":
    main()
