#!/bin/bash
# One-shot progress report for a Phase 2 ablation array job.
#
# Usage:
#   ./scripts/ablation_status.sh <jobid>
#   ./scripts/ablation_status.sh             # auto-detects most recent ablation array
#
# Shows: state counts, sample completed val losses, failures, ETA estimate.

set -euo pipefail

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
    # Auto-detect: most recent job named 'ablation' for this user.
    JOB=$(sacct --user="$USER" --name=ablation -X --noheader --format=JobID --starttime=now-3days \
        | awk -F'_' '{print $1}' | sort -u | tail -1)
    if [[ -z "$JOB" ]]; then
        echo "usage: $0 <jobid>   (no recent ablation job found to auto-detect)" >&2
        exit 1
    fi
    echo "Auto-detected job: $JOB"
fi

CONCURRENT=24
LOG_DIR="results/slurm"

# Auto-detect the array size from scontrol (handles redo arrays where the
# total isn't 250). Spec looks like "ArrayTaskId=0-249%24" for the main array
# or "ArrayTaskId=24,25,27-32,..." for explicit-index redos. Fall back to
# sacct row count if scontrol can't see the job (e.g., already finished and
# purged from the controller), then to 250 as a final fallback.
detect_total() {
    local spec n
    spec=$(scontrol show job "$1" 2>/dev/null \
        | grep -oE 'ArrayTaskId=[0-9,%-]+' | head -1 \
        | sed 's/^ArrayTaskId=//' | sed 's/%.*//')
    if [[ -n "$spec" ]]; then
        n=$(echo "$spec" | tr ',' '\n' | awk -F'-' '
            NF==1 {n++}
            NF==2 {n += $2 - $1 + 1}
            END {print n+0}')
        if [[ -n "$n" && "$n" -gt 0 ]]; then echo "$n"; return; fi
    fi
    n=$(sacct -j "$1" -X --noheader --format=JobID 2>/dev/null | wc -l | tr -d ' ')
    if [[ -n "$n" && "$n" -gt 0 ]]; then echo "$n"; return; fi
    echo 250
}
TOTAL=$(detect_total "$JOB")

# ---------------------------------------------------------------------------
# 1. State counts
# ---------------------------------------------------------------------------
echo ""
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') — Job $JOB state counts (size=$TOTAL) ==="

# Fetch all states once. sacct's --state filter has version-quirks (returns
# spurious rows on some SLURM builds), so we pull the full state column and
# count in awk to be safe.
STATES=$(sacct -j "$JOB" -X --noheader --format=State%-20 2>/dev/null \
    | awk '{gsub(/[+ ]+$/, "", $0); print $1}')
echo "$STATES" | sort | uniq -c

count_state() {
    # Match any of the comma-separated state names passed in.
    echo "$STATES" | awk -v pats="$1" '
        BEGIN {n = split(pats, a, ","); for (i=1;i<=n;i++) wanted[toupper(a[i])]=1}
        toupper($1) in wanted {c++}
        END {print c+0}'
}
N_COMPLETED=$(count_state COMPLETED)
N_RUNNING=$(count_state RUNNING)
N_PENDING=$(count_state PENDING)
N_FAILED=$(count_state FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED,NODE_FAIL)

# ---------------------------------------------------------------------------
# 2. Sample completed val losses (last 5 to finish)
# ---------------------------------------------------------------------------
echo ""
echo "=== Sample completed val losses (last 5 to finish) ==="
if [[ "$N_COMPLETED" -eq 0 ]]; then
    echo "  (no completed tasks yet)"
else
    sacct -j "$JOB" -X --state=COMPLETED --noheader --format=JobID,End \
        | sort -k2 | tail -5 | awk '{print $1}' | while read -r jid; do
        out="$LOG_DIR/ablation-${jid}.out"
        if [[ -f "$out" ]]; then
            loss=$(grep "val_loss" "$out" | tail -1 | grep -oE 'val_loss:[0-9.]+' | cut -d: -f2)
            cfg=$(grep -oE 'cfg=name=[^ ]+' "$out" | head -1 | sed 's/cfg=name=//')
            printf "  %-18s  %-48s  final_val_loss=%s\n" "$jid" "${cfg:-?}" "${loss:-?}"
        else
            printf "  %-18s  (no .out file at %s)\n" "$jid" "$out"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 3. Failures
# ---------------------------------------------------------------------------
echo ""
echo "=== Failures ($N_FAILED total) ==="
if [[ "$N_FAILED" -eq 0 ]]; then
    echo "  (none)"
else
    sacct -j "$JOB" -X --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED,NODE_FAIL \
        --format=JobID%-18,State%-12,ExitCode,Elapsed | head -20
    if [[ "$N_FAILED" -gt 18 ]]; then
        echo "  ... (showing first 18 of $N_FAILED)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. ETA estimate
# ---------------------------------------------------------------------------
echo ""
echo "=== ETA estimate ==="
if [[ "$N_COMPLETED" -lt 3 ]]; then
    echo "  (need ≥3 completed tasks for a reliable median; have $N_COMPLETED)"
else
    # Median elapsed time across completed tasks, in seconds.
    MEDIAN_SEC=$(sacct -j "$JOB" -X --state=COMPLETED --noheader --format=Elapsed \
        | awk -F: 'NF==3 {print $1*3600+$2*60+$3}' \
        | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}')

    REMAINING=$(( TOTAL - N_COMPLETED - N_FAILED ))
    # ceil(REMAINING / CONCURRENT) batches.
    BATCHES=$(( (REMAINING + CONCURRENT - 1) / CONCURRENT ))
    ETA_SEC=$(( BATCHES * MEDIAN_SEC ))

    printf "  median elapsed/task : %dm %ds\n" $(( MEDIAN_SEC / 60 )) $(( MEDIAN_SEC % 60 ))
    printf "  completed           : %d / %d (%.0f%%)\n" "$N_COMPLETED" "$TOTAL" \
        "$(awk "BEGIN {print $N_COMPLETED/$TOTAL*100}")"
    printf "  running / pending   : %d / %d\n" "$N_RUNNING" "$N_PENDING"
    printf "  failed (excl.)      : %d\n" "$N_FAILED"
    printf "  est. remaining time : %dh %dm  (%d batches × ~%dm)\n" \
        $(( ETA_SEC / 3600 )) $(( (ETA_SEC % 3600) / 60 )) \
        "$BATCHES" $(( MEDIAN_SEC / 60 ))
fi

echo ""
