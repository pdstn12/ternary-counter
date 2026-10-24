"""Paper figures 1-3 from the saved runs (Colab zip) and the outline tables. Usage: python make_figures.py RUNS_DIR OUT_DIR"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
# one colour per entity, the same in every figure (validated categorical slots 1-6)
C = dict(counter='#2a78d6', qat='#eb6834', fixed='#1baf7a', linear='#eda100', sr='#e87ba4', lta='#008300',
         ref='#6b6a64', ink='#1f1f1d', muted='#8a897f')
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': C['muted'], 'axes.linewidth': 0.6,
                     'axes.labelcolor': C['ink'], 'xtick.color': C['muted'], 'ytick.color': C['muted'],
                     'xtick.labelcolor': C['ink'], 'ytick.labelcolor': C['ink'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def load(name):
    return json.load(open(f'{RUNS}/{name}/result.json'))


def log(name):
    return [json.loads(l) for l in open(f'{RUNS}/{name}/log.jsonl')]


def curve(name):
    x, y = zip(*load(name)['curve'])
    return np.array(x) / 1e6, np.array(y)


def grid(ax):
    ax.grid(axis='y', color='#e4e3dc', lw=0.5); ax.set_axisbelow(True)


def end_labels(ax, items, gap):
    """value labels at the right end of each line, pushed apart vertically so they never overlap"""
    items = sorted(items, key=lambda t: t[1])
    ys = [y for _, y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, y, txt), yl in zip(items, ys):
        ax.annotate(txt, (x, y), xytext=(x * 1.02 + 2, yl), textcoords='data', va='center', fontsize=7.5, color=C['ink'])


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300)
    plt.close(fig)


# ---------- Figure 2: validation loss from scratch ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]
ends = []
series = [  # label, colour, runs (mean if several, band = min..max), style
    ('fp (reference)', C['ref'], ['main_fp'], dict(ls='--', marker=None)),
    ('QAT', C['qat'], ['main_qat'], dict(ls='-', marker='s')),
    ('counter, adaptive τ', C['counter'], ['a100_main_counter_adapt_8_64', 'a100_ctrl_adapt_8_64_s1'], dict(ls='-', marker='o')),
    ('counter, linear τ', C['linear'], ['a100_ctrl_linear_8_26_s0'], dict(ls='-', marker='^')),
    ('counter, fixed τ', C['fixed'], ['a100_main_counter_fixed_t8', 'a100_ctrl_fixed_t8_s1'], dict(ls='-', marker='D')),
    ('stochastic rounding', C['sr'], ['main_sr'], dict(ls='-', marker='v')),
]
for label, col, runs, st in series:
    cs = [curve(r) for r in runs]
    x, ys = cs[0][0], np.array([c[1] for c in cs])
    ax.plot(x, ys.mean(0), color=col, lw=1.6, ms=3.5, label=label, **st)
    if len(runs) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=col, alpha=0.18, lw=0)
    ends.append((x[-1], ys.mean(0)[-1], f'{ys.mean(0)[-1]:.2f}'))
end_labels(ax, ends, 0.1)
ax.set_title('a) 51M parameters, 200M tokens', loc='left', fontsize=9, color=C['ink'])
ax.set_xlabel('training tokens (M)'); ax.set_ylabel('validation loss (nats)')
ax.set_ylim(1.3, 3.2); ax.set_xlim(0, 228); grid(ax)
ax = axes[1]
ends = []
for label, col, run, st in [('QAT', C['qat'], 'gpt2s_qat', dict(marker='s')),
                            ('counter, adaptive τ', C['counter'], 'gpt2s_counter_adapt_8_64', dict(marker='o')),
                            ('counter, fixed τ', C['fixed'], 'gpt2s_counter_fixed_t8', dict(marker='D'))]:
    x, y = curve(run)
    ax.plot(x, y, color=col, lw=1.6, ms=3.5, label=label, **st)
    ends.append((x[-1], y[-1], f'{y[-1]:.2f}'))
end_labels(ax, ends, 0.1)
ax.set_title('b) 124M parameters, 150M tokens (1 seed)', loc='left', fontsize=9, color=C['ink'])
ax.set_xlabel('training tokens (M)'); ax.set_xlim(0, 172); grid(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=7.5, handlelength=2.2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, 'fig2_from_scratch_loss')

# ---------- Figure 3: tau and cumulative flips ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
tok_per_step = 32768
ax = axes[0]
for run, ls, lab in [('a100_main_counter_adapt_8_64', '-', 'adaptive τ, seed 0'), ('a100_ctrl_adapt_8_64_s1', ':', 'adaptive τ, seed 1')]:
    tr = load(run)['tau_trace']
    s, t = zip(*[(a[0], a[1]) for a in tr])
    ax.step(np.array(s) * tok_per_step / 1e6, t, where='post', color=C['counter'], ls=ls, lw=1.4, label=lab)
steps = load('a100_ctrl_linear_8_26_s0')['steps']
ax.plot([0, steps * tok_per_step / 1e6], [8, 26], color=C['linear'], lw=1.4, label='linear τ 8→26')
ax.plot([0, steps * tok_per_step / 1e6], [8, 8], color=C['fixed'], lw=1.4, label='fixed τ 8')
ax.set_title('a) threshold τ', loc='left', fontsize=9, color=C['ink'])
ax.set_xlabel('training tokens (M)'); ax.set_ylabel('τ'); ax.legend(fontsize=7.5); grid(ax)
ax = axes[1]
for run, col, ls, lab in [('a100_main_counter_adapt_8_64', C['counter'], '-', 'adaptive τ'),
                          ('a100_ctrl_linear_8_26_s0', C['linear'], '-', 'linear τ'),
                          ('a100_main_counter_fixed_t8', C['fixed'], '-', 'fixed τ'),
                          ('main_sr', C['sr'], '--', 'stochastic rounding')]:
    lg = [r for r in log(run) if 'flips' in r]
    x = np.array([r['tokens'] for r in lg]) / 1e6
    y = np.array([r['flips'] for r in lg])
    ax.plot(x, y, color=col, ls=ls, lw=1.4, label=lab)
    ax.annotate(f'{y[-1]:.1f}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7.5, color=C['ink'])
ax.set_title('b) cumulative flips per weight', loc='left', fontsize=9, color=C['ink'])
ax.set_xlabel('training tokens (M)'); ax.set_ylabel('flips per weight'); ax.set_xlim(0, 225); ax.legend(fontsize=7.5); grid(ax)
fig.tight_layout()
save(fig, 'fig3_tau_and_flips')

# ---------- Figure 1: task score vs forgetting (numbers from Tables 1-2 of the outline) ----------
panels = [
    ('a) TriLM 560M, RTE', 'accuracy (%)', 26.16, 63.2, [
        ('counter', C['counter'], 'o', 63.4, 4.6, 29.42),
        ('stochastic rounding', C['sr'], 'v', 64.7, 3.3, 41.66),
        ('Low-Rank Ternary Adapt.', C['lta'], 'P', 61.1, 5.8, 52.76),
        ('zero-shot = merged LoRA', C['ref'], 'X', 51.3, 0, 26.16)],
     ('QAT, best of 10', 50.8, 1.5, 11153)),
    ('b) BitNet b1.58 2B, Tunisian sentiment', 'macro-F1 (%)', 20.09, 72.4, [
        ('counter', C['counter'], 'o', 72.2, 1.1, 20.74),
        ('stochastic rounding', C['sr'], 'v', 72.8, 1.1, 21.51),
        ('Low-Rank Ternary Adapt.', C['lta'], 'P', 72.9, 2.3, 84.72),
        ('QAT (bf16 + 8-bit AdamW)', C['qat'], 's', 61.3, 2.8, 20.09),
        ('zero-shot = merged LoRA', C['ref'], 'X', 26.8, 0, 20.09)],
     None),
]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, (title, ylab, ppl0, lora, pts, offscale) in zip(axes, panels):
    ax.axhline(lora, color=C['muted'], ls=':', lw=1)
    ax.annotate('LoRA adapter (not ternary)', (0.02, lora), xycoords=('axes fraction', 'data'), xytext=(0, -4), va='top',
                textcoords='offset points', ha='left', fontsize=7, color=C['muted'])
    for lab, col, mk, y, e, ppl in pts:
        x = 100 * (ppl / ppl0 - 1)
        ax.errorbar(x, y, yerr=e if e else None, fmt=mk, color=col, ms=6.5, mec='white', mew=0.8,
                    elinewidth=1.2, capsize=2.5, label=lab, zorder=3)
    if offscale:
        lab, y, e, ppl = offscale
        ax.annotate(f'{lab}: {y} ± {e},\nWikiText ppl {ppl:,} (+{100 * (ppl / ppl0 - 1):,.0f}%), off scale',
                    (0.98, 0.04), xycoords='axes fraction', ha='right', va='bottom', fontsize=7, color=C['qat'])
    ax.set_xscale('symlog', linthresh=5); ax.set_xlim(-1, 500)
    ax.set_xticks([0, 5, 10, 50, 100, 300]); ax.set_xticklabels(['0', '5', '10', '50', '100', '300'])
    ax.set_title(title, loc='left', fontsize=9, color=C['ink'])
    ax.set_xlabel('WikiText-2 perplexity increase vs zero-shot (%)'); ax.set_ylabel(ylab); grid(ax)
hs, ls_ = {}, {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        hs.setdefault(l, h)
fig.legend(list(hs.values()), list(hs.keys()), loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, 'fig1_score_vs_forgetting')
print('ok', sorted(os.listdir(OUT)))

# ---------- Figure 4: fine-tuning peak memory vs GPU capacity ----------
fig, ax = plt.subplots(figsize=(7.0, 2.6))
rows = [  # label, value GB, colour, hatch, note
    ('TriLM 560M · counter', 2.2, C['counter'], None, '2.2 GB'),
    ('TriLM 560M · QAT (fp32 + AdamW)', 9.1, C['qat'], None, '9.1 GB'),
    ('BitNet 2B · counter', 6.5, C['counter'], None, '6.5 GB'),
    ('BitNet 2B · QAT (bf16 + 8-bit AdamW)', 15.7, '#f4b49a', None, '15.7 GB'),
    ('BitNet 2B · QAT (fp32 + AdamW)', 40.0, 'white', '////', 'out of memory (38.9 GB allocated at the first step)'),
]
y = np.arange(len(rows))[::-1].astype(float)
y[:2] += 0.4                                     # gap between the two models
for (lab, v, col, hatch, note), yy in zip(rows, y):
    ec = C['qat'] if col in ('white', C['qat'], '#f4b49a') else col
    ax.barh(yy, v, height=0.62, color=col, edgecolor=ec, hatch=hatch, lw=1.0)
    ax.text(v + 0.6 if v < 40 else 28, yy, note, va='center', ha='left' if v < 40 else 'center', fontsize=7.5,
            color=C['ink'], bbox=None if v < 40 else dict(facecolor='white', edgecolor='none', pad=1.5))
for x, lab in ((16, 'T4 · 16 GB'), (40, 'A100 · 40 GB')):
    ax.axvline(x, color=C['muted'], ls='--', lw=0.9)
    ax.text(x - 0.5, y.max() + 0.75, lab, ha='right', va='bottom', fontsize=7.5, color=C['muted'])
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 44); ax.set_ylim(y.min() - 0.6, y.max() + 1.2)
ax.set_xlabel('peak GPU memory during fine-tuning (GB)')
ax.grid(axis='x', color='#e4e3dc', lw=0.5); ax.set_axisbelow(True)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
fig.tight_layout()
save(fig, 'fig4_finetune_memory')
print('fig4 ok')
