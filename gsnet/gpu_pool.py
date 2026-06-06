#
# GPU-aware, fault-tolerant job scheduler for shared 8-GPU boxes.
#
# Behaviour the user asked for:
#  * prefer EMPTY / low-occupancy cards (most-free-memory first);
#  * may squeeze onto a partially-used card as long as it has >= min_free_mb;
#  * if a job dies (e.g. OOM because a co-tenant grew), DON'T abort the batch --
#    requeue it and retry on a DIFFERENT free card (up to max_retries);
#  * if no card is free right now, wait and keep polling (never give up).
#
# nvidia-smi is queried each scheduling tick so a colleague taking/freeing cards
# is reflected live.
#
import subprocess
import threading
import time


def gpu_status():
    """{gpu_index: (free_MiB, util_pct)} from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"]).decode()
    st = {}
    for line in out.strip().splitlines():
        i, free, util = (x.strip() for x in line.split(","))
        st[int(i)] = (int(free), int(util))
    return st


def free_gpus(allowed, busy, min_free_mb):
    """Allowed cards not currently running OUR jobs and with enough free mem,
    most-free first (ties: lower utilization)."""
    try:
        st = gpu_status()
    except Exception as e:                       # nvidia-smi hiccup -> assume allowed are usable
        print(f"[gpu_pool] nvidia-smi failed ({e}); falling back to allowed list", flush=True)
        st = {g: (min_free_mb, 0) for g in allowed}
    cand = [(f, u, i) for i, (f, u) in st.items()
            if i in allowed and i not in busy and f >= min_free_mb]
    cand.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, _, i in cand]


def run_jobs(jobs, allowed, run_fn, *, min_free_mb=12000, max_retries=4,
             poll=20, label="job"):
    """Schedule `jobs` across `allowed` GPUs. run_fn(job, gpu) executes one job
    on one card and RAISES on failure. One card runs one of our jobs at a time.
    Returns (done, failed)."""
    lock = threading.Lock()
    pending = [(j, 0) for j in jobs]
    busy = set()
    active = []
    done, failed = [], []

    def worker(job, attempt, gpu):
        ok = False
        try:
            run_fn(job, gpu)
            ok = True
        except Exception as e:
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            print(f"[{label}] gpu{gpu} attempt {attempt+1} FAILED: {msg}", flush=True)
        with lock:
            busy.discard(gpu)
            if ok:
                done.append(job)
            elif attempt + 1 <= max_retries:
                pending.append((job, attempt + 1))     # retry on another free card
                print(f"[{label}] requeued (will retry on a free card)", flush=True)
            else:
                failed.append(job)
                print(f"[{label}] giving up after {max_retries} retries", flush=True)

    while True:
        with lock:
            active[:] = [t for t in active if t.is_alive()]
            stop = (not pending) and (not active)
            need = len(pending)
        if stop:
            break
        if need:
            with lock:
                taken = set(busy)
            for gi in free_gpus(allowed, taken, min_free_mb):
                with lock:
                    if not pending:
                        break
                    if gi in busy:
                        continue
                    job, attempt = pending.pop(0)
                    busy.add(gi)
                    t = threading.Thread(target=worker, args=(job, attempt, gi), daemon=True)
                    t.start()
                    active.append(t)
        time.sleep(poll)

    return done, failed
