"""Frontier plot: every tuning setting of every method on BitNet 2B / Tunisian (500 held-out comments),
task score vs SST-2 kept (same 500 held-out SST-2 sentences for all runs; unchanged model = 0.870)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = dict(counter='#2a78d6', qat='#eb6834', sr='#e87ba4', lta='#008300', ref='#6b6a64', ink='#1f1f1d', muted='#8a897f')
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.edgecolor': C['muted'], 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
# (macro-F1, SST-2) from the tuning logs
counter = [(70.1, .892), (68.5, .926), (75.3, .880), (73.1, .848), (72.5, .816), (71.6, .792), (70.1, .878), (68.6, .830),
           (67.5, .812), (67.4, .594), (66.4, .658), (65.8, .734), (65.2, .820), (63.8, .666), (63.3, .848), (36.0, .578)]
sr = [(62.1, .760), (66.1, .856), (73.1, .876), (69.5, .774)]
lta = [(52.8, .578), (70.3, .578), (51.9, .784), (32.4, .870), (73.4, .592), (61.8, .694), (32.4, .870)]
qat = [(36.0, .578), (56.4, .798), (32.4, .870), (46.7, .578), (32.8, .862), (32.8, .866)]
lora = [(75.5, .890), (71.2, .896), (66.5, .754)]


def frontier(pts):
    best, out = -1, []
    for x, y in sorted(pts, key=lambda p: -p[0]):      # from the highest score down, keep points that retain more
        if y > best:
            out.append((x, y)); best = y
    return sorted(out)


fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.axhline(.870, color=C['muted'], ls=':', lw=1)
ax.text(38, .874, 'unchanged model (0.870)', fontsize=7, color=C['muted'])
ax.axhline(.89, color=C['muted'], ls='--', lw=0.8)
ax.text(38, .894, 'selection rule: SST-2 ≥ 0.89', fontsize=7, color=C['muted'])
for name, pts, col, mk in (('counter (16 settings)', counter, C['counter'], 'o'), ('stochastic rounding (4)', sr, C['sr'], 'v'),
                           ('Low-Rank Ternary Adapt. (7)', lta, C['lta'], 'P'), ('QAT, bf16 + 8-bit (6)', qat, C['qat'], 's')):
    x, y = zip(*pts)
    ax.scatter(x, y, color=col, marker=mk, s=34, edgecolor='white', lw=0.6, label=name, zorder=3)
    fx, fy = zip(*frontier(pts))
    ax.plot(fx, fy, color=col, lw=1.4, zorder=2)
x, y = zip(*lora)
ax.scatter(x, y, facecolor='none', edgecolor=C['ref'], marker='D', s=30, label='LoRA adapter (not ternary, 3)', zorder=3)
ax.set_xlabel('task score: macro-F1 on 500 held-out Tunisian comments')
ax.set_ylabel('SST-2 accuracy kept (500 held-out sentences)')
ax.set_xlim(30, 78); ax.set_ylim(.56, .94)
ax.grid(color='#e4e3dc', lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
ax.set_title('Every tuning setting, BitNet 2B · Tunisian (upper right is better; lines = each method\'s frontier)',
             loc='left', fontsize=8.5, color=C['ink'])
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/fig5_frontier_tunisian.{ext}', dpi=300)
print('ok')
