#!/usr/bin/env bash
# scripts/simp_dashboard.sh
#
# Full monitoring dashboard for fenics-pipeline SIMP runs.
#
# Layout:
#   ┌─────────────────────────────────────┬──────────────────────────┐
#   │  Pane 0 — papermill live output     │  Pane 1 — GPU monitor    │
#   │           (full solver stdout)      │           nvidia-smi     │
#   │                                     ├──────────────────────────┤
#   │                                     │  Pane 3 — result.json    │
#   ├─────────────────────────────────────┤           live status    │
#   │  Pane 2 — Iter monitor              │                          │
#   │           ETA · sparkline · CG      │                          │
#   └─────────────────────────────────────┴──────────────────────────┘
#
# Usage (from repo root in WSL2):
#   bash scripts/simp_dashboard.sh [cpu|auto] [filter_radius] [vf] [max_iter]
#
# Defaults: cpu  4.5  0.30  200
# Detach:   Ctrl+B, D     Re-attach: tmux attach -t simp_run

set -euo pipefail

BACKEND="${1:-cpu}"
FILTER_R="${2:-4.5}"
VF="${3:-0.30}"
MAX_ITER="${4:-200}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="$REPO/outputs/problem/run.log"
RESULT_PATH="$REPO/outputs/problem/result.json"
SESSION="simp_run"

mkdir -p "$REPO/outputs/problem"
touch "$LOG_PATH"
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Write the Python iter monitor to a temp file ──────────────────────────────
MONITOR_PY="/tmp/simp_monitor_$$.py"
cat > "$MONITOR_PY" << 'PYEOF'
#!/usr/bin/env python3
"""
Live iter monitor for simp_dashboard.sh
Reads the papermill log, parses Iter lines, shows ETA + sparkline + CG trend.
"""
import sys, time, re, os
from datetime import timedelta

LOG  = sys.argv[1]
MXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 200

C = {
    'R': '\033[91m', 'G': '\033[92m', 'Y': '\033[93m',
    'B': '\033[94m', 'M': '\033[95m', 'C': '\033[96m',
    'W': '\033[97m', 'bold': '\033[1m', 'dim': '\033[2m',
    'X': '\033[0m',
}
def c(t, *codes): return ''.join(C.get(k,'') for k in codes) + t + C['X']

def sparkline(vals, w=44):
    if len(vals) < 2: return c('─'*w, 'dim')
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1e-30
    chars = ' ▁▂▃▄▅▆▇█'
    sub = vals[-w:]
    out = ''
    for v in sub:
        idx = int((v - mn) / rng * 8)
        idx = max(0, min(8, idx))
        col = 'G' if idx < 3 else ('Y' if idx < 6 else 'R')
        out += c(chars[idx], col)
    return out

def bar(cur, tot, w=36):
    pct = min(cur / max(tot, 1), 1.0)
    filled = int(pct * w)
    b = c('█' * filled, 'G') + c('░' * (w - filled), 'dim')
    return f'[{b}] {c(f"{pct*100:5.1f}%", "bold")}'

def eta(elapsed, cur, tot):
    if cur < 2: return c('estimating…', 'dim')
    rate = elapsed / cur
    rem  = (tot - cur) * rate
    return str(timedelta(seconds=int(rem)))

iters, comps, cgs, t_start = [], [], [], None
last_backend = ''
last_stage   = ''

def redraw():
    os.system('clear')
    w = 56
    print(c(' ╔' + '═'*(w-2) + '╗', 'C'))
    print(c(' ║', 'C') + c(f'  SIMP MONITOR — fenics-pipeline'.center(w-2), 'bold','W') + c('║', 'C'))
    print(c(' ╚' + '═'*(w-2) + '╝', 'C'))
    if last_backend: print(c(f'  ▶ {last_backend}', 'C'))
    if last_stage:   print(c(f'  ◉ {last_stage}', 'M', 'bold'))
    print()

    cur = iters[-1] if iters else 0
    elapsed = time.time() - t_start if t_start else 0
    print(f'  Progress  {bar(cur, MXIT)}')
    print(f'  Iter      {c(str(cur), "bold","W")} / {MXIT}   '
          f'Elapsed {c(str(timedelta(seconds=int(elapsed))), "Y")}   '
          f'ETA {c(eta(elapsed, cur, MXIT), "Y")}')
    print()

    if comps:
        last_c = comps[-1]
        col = 'G' if last_c < 0.06 else ('Y' if last_c < 0.10 else 'W')
        print(f'  Compliance  {c(f"{last_c:.6e}", col, "bold")}', end='')
        if len(comps) > 1:
            d = comps[-1] - comps[-2]
            arrow = c('↓ converging', 'G') if d < 0 else c('↑ check!', 'R')
            print(f'   Δ {d:+.3e}  {arrow}')
        else:
            print()
        last_cg = cgs[-1] if cgs else '?'
        cg_col = 'G' if isinstance(last_cg, int) and last_cg < 80 else 'Y'
        print(f'  CG iters    {c(str(last_cg), cg_col)}  '
              + c('(AMG working)', 'G', 'dim') if isinstance(last_cg,int) and last_cg<100 else '')
        print()
        print(f'  Compliance trend (low=good)')
        print(f'  {sparkline(comps)}')
        print(f'  {c(f"{comps[0]:.3e}", "dim")}{"─"*38}{c(f"{comps[-1]:.3e}", "dim")}')
        print()

    if len(iters) > 1:
        print(c(f'  {"Iter":>5}  {"Compliance":>12}  {"CG":>5}  {"Δ":>10}', 'dim'))
        show = list(zip(iters, comps, cgs))[-12:]
        for i, (it, cp, cg) in enumerate(show):
            delta = ''
            if i > 0:
                d = cp - show[i-1][1]
                delta = c(f'{d:+.3e}', 'G' if d < 0 else 'R')
            cg_s = c(str(cg), 'G' if cg < 80 else 'Y')
            print(f'  {it:>5}  {cp:>12.6e}  {cg_s:>5}  {delta}')

with open(LOG, 'r') as f:
    f.seek(0)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.3)
            continue

        if 'Solver backend' in line or 'AMGCL' in line:
            last_backend = line.strip()
            redraw()

        elif re.search(r'STAGE|Stage [12]', line):
            last_stage = line.strip()
            redraw()

        else:
            m = re.search(
                r'Iter\s+(\d+)\s*\|.*?C=([\d.e+\-]+).*?'
                r'CG=\s*(\d+).*?([\d.]+)s', line)
            if m:
                it  = int(m.group(1))
                cp  = float(m.group(2))
                cg  = int(m.group(3))
                sec = float(m.group(4))
                if t_start is None:
                    t_start = time.time() - sec
                iters.append(it); comps.append(cp); cgs.append(cg)
                redraw()
PYEOF

# ── Write result.json watcher ─────────────────────────────────────────────────
RESULT_PY="/tmp/simp_result_$$.py"
cat > "$RESULT_PY" << 'PYEOF'
#!/usr/bin/env python3
import sys, time, json, os
from datetime import timedelta

PATH = sys.argv[1]
C = {
    'G': '\033[92m', 'Y': '\033[93m', 'R': '\033[91m',
    'C': '\033[96m', 'bold': '\033[1m', 'dim': '\033[2m', 'X': '\033[0m'
}
def c(t, *k): return ''.join(C.get(x,'') for x in k) + t + C['X']

def chart(vals, w=24, h=8):
    """Tiny ASCII compliance chart"""
    if len(vals) < 2: return []
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1e-30
    sub = vals[-w:]
    rows = []
    for row in range(h, 0, -1):
        thresh = mn + rng * row / h
        line = ''
        for v in sub:
            line += c('█', 'G') if v <= thresh else ' '
        rows.append(line)
    return rows

while True:
    os.system('clear')
    print(c(' ╔══════════════════════════╗', 'C'))
    print(c(' ║  CONVERGENCE STATUS      ║', 'C', 'bold'))
    print(c(' ╚══════════════════════════╝', 'C'))
    print()

    if not os.path.exists(PATH):
        print(c('  ⏳ Awaiting result.json…', 'Y'))
        print(c('  Run has not produced output yet.', 'dim'))
    else:
        try:
            d = json.load(open(PATH))
            h = d.get('compliance_history', [])
            dur = d.get('duration_s', 0)
            conv = d['converged']
            status_s = c('✓ CONVERGED', 'G', 'bold') if conv else c('⏳ running…', 'Y')
            print(f'  Status   {status_s}')
            print(f'  Iters    {c(str(d["n_iterations"]), "bold")}')
            print(f'  Final C  {c(f"{d[\"final_compliance\"]:.4e}", "G", "bold")}')
            print(f'  Vol frac {d.get("final_volume_frac", "?"):.4f}')
            print(f'  Duration {str(timedelta(seconds=int(dur)))}')
            print()
            if len(h) >= 2:
                rows = chart(h, w=24, h=7)
                mn, mx = min(h), max(h)
                print(c(f'  Compliance history  ({len(h)} iters)', 'dim'))
                print(c(f'  {mx:.3e} ┐', 'dim'))
                for row in rows:
                    print(f'           │{row}│')
                print(c(f'  {mn:.3e} └{"─"*24}┘', 'dim'))
                print()
                print(c('  Last 5:', 'dim'))
                for i, v in enumerate(h[-5:]):
                    bar_w = int((h[-1] / v * 20)) if v > 0 else 20
                    print(f'  {len(h)-4+i:4d} | {v:.4e}  '
                          + c('█'*min(bar_w,20), 'G'))
        except Exception as e:
            print(c(f'  Error reading result.json: {e}', 'R'))
    time.sleep(4)
PYEOF

# ── Build commands ─────────────────────────────────────────────────────────────
# FIX: wrap in bash -c so docker exec doesn't consume the -p flags
RUN_CMD="docker-compose exec fenics-pipeline bash -c \
'cd /workspace && papermill \
  notebooks/04_simp_optimization.ipynb /tmp/nb04_out.ipynb \
  -p SOLVER_BACKEND ${BACKEND} \
  -p FILTER_RADIUS ${FILTER_R} \
  -p VOLUME_FRACTION ${VF} \
  -p MAX_CG_ITER ${MAX_ITER} \
  --log-output 2>&1 | tee outputs/problem/run.log'"

SMI_CMD="watch -n 2 \"\
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,\
temperature.gpu,power.draw,clocks.current.sm \
  --format=csv,noheader,nounits \
  | awk -F', ' '{
      printf \\\" ╔══════════════════════════╗\n\\\";
      printf \\\" ║  GPU MONITOR             ║\n\\\";
      printf \\\" ╚══════════════════════════╝\n\n\\\";
      printf \\\" %-24s\n\n\\\",\\\$1;
      printf \\\"  GPU  util : %3s%%\n\\\",\\\$2;
      printf \\\"  Mem  util : %s / %sMB\n\\\",\\\$3,\\\$4;
      printf \\\"  Temp      : %s C\n\\\",\\\$5;
      printf \\\"  Power     : %s W\n\\\",\\\$6;
      printf \\\"  SM clock  : %s MHz\n\\\",\\\$7;
    }'\""

MON_CMD="python3 $MONITOR_PY '$LOG_PATH' $MAX_ITER"
RES_CMD="python3 $RESULT_PY '$RESULT_PATH'"

# ── Launch tmux ───────────────────────────────────────────────────────────────
COLS=$(tput cols  2>/dev/null || echo 200)
ROWS=$(tput lines 2>/dev/null || echo 52)
RIGHT=58   # width of right column
BOT=20     # height of bottom-left pane

tmux new-session -d -s "$SESSION" -x "$COLS" -y "$ROWS"

# Pane 0 top-left: run
tmux send-keys -t "$SESSION:0.0" "$RUN_CMD" Enter

# Pane 1 top-right: GPU
tmux split-window -t "$SESSION:0.0" -h -l "$RIGHT"
tmux send-keys    -t "$SESSION:0.1" "$SMI_CMD" Enter

# Pane 2 bottom-left: iter monitor
tmux select-pane  -t "$SESSION:0.0"
tmux split-window -t "$SESSION:0.0" -v -l "$BOT"
tmux send-keys    -t "$SESSION:0.2" "$MON_CMD" Enter

# Pane 3 bottom-right: result.json
tmux select-pane  -t "$SESSION:0.1"
tmux split-window -t "$SESSION:0.1" -v -l "$BOT"
tmux send-keys    -t "$SESSION:0.3" "$RES_CMD" Enter

tmux select-pane -t "$SESSION:0.0"

cat << INFO

  ┌─ simp_dashboard ─────────────────────────────────────────┐
  │  Session  : $SESSION
  │  Backend  : $BACKEND   Filter: ${FILTER_R}mm   VF: $VF   MaxIter: $MAX_ITER
  │
  │  tmux attach -t $SESSION        re-attach
  │  Ctrl+B, D                      detach (run keeps going)
  │  Ctrl+B, arrows / click         move between panes
  │  Ctrl+B, z                      zoom current pane fullscreen
  └───────────────────────────────────────────────────────────┘

INFO

tmux attach-session -t "$SESSION"