"""학습 중 검증 loss 가 가장 낮은 step_* 체크포인트를 runs/<run>/best 로 보존한다(하드링크라 추가 용량은 트레이너가 원본을 지운 뒤에만 ≤ 체크포인트 1개).

  train_b.py 는 --keep 개의 step_* 만 남기므로 중간의 최저 검증 체크포인트가 지워진다. 이 스크립트를 학습과 나란히 띄워 두면
  log.jsonl 의 val_loss 최저 업데이트의 step_ 폴더를 best 로 복사(cp -al)한다. best/VAL 에 업데이트 번호와 값을 적는다.
  트레이너가 사라지면 마지막 한 번 더 보고 끝난다. 여유 디스크 30 GB 미만이면 보존하지 않는다.
  트레이너 생존은 /proc 로 본다: python 인터프리터가 train_b.py 를 스크립트 인자로 가진 프로세스(좀비는 cmdline 이 비어 제외).
  --pid 를 주면 그 PID 만 본다(PID 가 재사용돼 다른 명령이 되면 죽은 것으로).

  nohup python keep_best.py ~/CSM/runs/b_emo1 [--pid <train_b.py 의 PID>] > ~/CSM/runs/b_emo1.best.log 2>&1 &
"""
import argparse, json, os, shutil, subprocess, time

MIN_FREE = 30 * 2**30
SCRIPT = b"train_b.py"


def trainer_alive(pid=None, proc="/proc"):
    """python …/train_b.py … 프로세스가 있으면 True. pgrep -f 는 'train_b.py' 문자열이 든 아무 명령(tail·grep·편집기)에나 걸려 쓰지 않는다."""
    if not os.path.isdir(proc): return False
    pids = [str(pid)] if pid else [p for p in os.listdir(proc) if p.isdigit()]
    for p in pids:
        try: args = open(os.path.join(proc, p, "cmdline"), "rb").read().split(b"\0")
        except OSError: continue
        if args and os.path.basename(args[0]).startswith(b"python") and any(os.path.basename(x) == SCRIPT for x in args[1:]):
            return True
    return False


def keep_once(run):
    """log.jsonl 의 val_loss 최저 step_ 을 best 로. 바뀌었으면 (업데이트, 값), 아니면 None."""
    best = os.path.join(run, "best"); tag = os.path.join(best, "VAL"); vals = []
    try:
        for line in open(os.path.join(run, "log.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if "val_loss" in r: vals.append((r["val_loss"], r["update"]))
    except FileNotFoundError:
        return None
    if not vals: return None
    v, u = min(vals); src = os.path.join(run, f"step_{u:06d}")
    cur = open(tag).read().split()[0] if os.path.exists(tag) else ""
    if cur == str(u) or not os.path.isdir(src) or os.path.exists(src + ".tmp"): return None
    if shutil.disk_usage(run).free < MIN_FREE:
        print(time.strftime("%H:%M"), "디스크 여유 부족 — 보존 안 함", flush=True); return None
    tmp = best + ".tmp"; shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["cp", "-al", src, tmp], check=True)
    open(os.path.join(tmp, "VAL"), "w").write(f"{u} {v}\n")
    shutil.rmtree(best, ignore_errors=True); os.rename(tmp, best)
    return u, v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--pid", type=int, default=None); ap.add_argument("--interval", type=float, default=60.0)
    a = ap.parse_args(); run = os.path.abspath(os.path.expanduser(a.run))
    while True:
        alive = trainer_alive(a.pid)
        r = keep_once(run)
        if r: print(time.strftime("%H:%M"), f"best → step_{r[0]:06d} (검증 {r[1]})", flush=True)
        if not alive:
            print(time.strftime("%H:%M"), "트레이너 없음 — 끝", flush=True); break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
