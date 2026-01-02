import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'


def load_results():
    """Load all benchmark_*.json files from output/ into a flat list."""
    results = []
    for jp in OUTPUT_DIR.glob('benchmark_*.json'):
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # attach filename as source
            for item in data:
                item.setdefault('source_file', jp.name)
            results.extend(data)
    return results


def group_by_profile_and_model(results):
    grouped = {}
    for r in results:
        profile = r.get('profile', 'unknown')
        model = r.get('model', 'unknown')
        key = (profile, model)
        grouped.setdefault(key, []).append(r)
    return grouped


def make_bar_plot_avg_fps(grouped):
    """One bar per (profile, model) with average FPS across videos."""
    profiles = sorted({p for (p, m) in grouped.keys()})
    models = sorted({m for (p, m) in grouped.keys()})

    # for кожного профілю окремий графік
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        xs = []
        ys = []
        for model in models:
            key = (profile, model)
            if key not in grouped:
                continue
            vals = grouped[key]
            avg_fps = sum(v['avg_fps'] for v in vals) / len(vals)
            xs.append(model)
            ys.append(avg_fps)

        if not xs:
            continue

        plt.figure(figsize=(10, 4))
        plt.bar(xs, ys)
        plt.ylabel('FPS (higher is better)')
        plt.title(f'Average FPS per model – profile: {profile}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_path = PLOTS_DIR / f'fps_{profile}.png'
        plt.savefig(out_path)
        plt.close()
        print(f'Saved {out_path}')


def make_bar_plot_avg_ms(grouped):
    """One bar per (profile, model) with average ms/frame across videos."""
    profiles = sorted({p for (p, m) in grouped.keys()})
    models = sorted({m for (p, m) in grouped.keys()})

    for profile in profiles:
        xs = []
        ys = []
        for model in models:
            key = (profile, model)
            if key not in grouped:
                continue
            vals = grouped[key]
            avg_ms = sum(v['avg_ms_per_frame'] for v in vals) / len(vals)
            xs.append(model)
            ys.append(avg_ms)

        if not xs:
            continue

        plt.figure(figsize=(10, 4))
        plt.bar(xs, ys)
        plt.ylabel('ms/frame (lower is better)')
        plt.title(f'Average latency per model – profile: {profile}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_path = PLOTS_DIR / f'latency_{profile}.png'
        plt.savefig(out_path)
        plt.close()
        print(f'Saved {out_path}')


def main():
    results = load_results()
    if not results:
        print('No benchmark_*.json files found in output/. Run run_pipelines.py first.')
        return

    grouped = group_by_profile_and_model(results)
    make_bar_plot_avg_fps(grouped)
    make_bar_plot_avg_ms(grouped)


if __name__ == '__main__':
    main()
