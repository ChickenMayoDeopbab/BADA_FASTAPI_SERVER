"""학습 중 검증 loss 가 가장 낮은 step_* 체크포인트를 runs/<run>/best 로 보존한다(하드링크라 추가 용량은 트레이너가 원본을 지운 뒤에만 ≤ 체크포인트 1개).

  train_b.py 는 --keep 개의 step_* 만 남기므로 중간의 최저 검증 체크포인트가 지워진다. 이 스크립트를 학습과 나란히 띄워 두면
  log.jsonl 의 val_loss 최저 업데이트의 step_ 폴더를 best 로 복사(cp -al)한다. best/VAL 에 업데이트 번호와 값을 적는다.
  트레이너(train_b.py)가 사라지면 마지막 한 번 더 보고 끝난다. 여유 디스크 30 GB 미만이면 보존하지 않는다.

  nohup python keep_best.py ~/CSM/runs/b_emo1 > ~/CSM/runs/b_emo1.best.log 2>&1 &
"""
import json, os, shutil, subprocess, sys, time

RUN = os.path.abspath(os.path.expanduser(sys.argv[1])); BEST = os.path.join(RUN, "best"); TAG = os.path.join(BEST, "VAL")
MIN_FREE = 30 * 2**30

def trainer_alive():
    return subprocess.run(["pgrep", "-f", "train_b.py"], capture_output=True).returncode == 0

while True:
    alive = trainer_alive()
    vals = []
    try:
        for line in open(os.path.join(RUN, "log.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if "val_loss" in r: vals.append((r["val_loss"], r["update"]))
    except FileNotFoundError:
        pass
    if vals:
        v, u = min(vals); src = os.path.join(RUN, f"step_{u:06d}")
        cur = open(TAG).read().split()[0] if os.path.exists(TAG) else ""
        if cur != str(u) and os.path.isdir(src) and not os.path.exists(src + ".tmp"):
            if shutil.disk_usage(RUN).free < MIN_FREE:
                print(time.strftime("%H:%M"), "디스크 여유 부족 — 보존 안 함", flush=True)
            else:
                tmp = BEST + ".tmp"; shutil.rmtree(tmp, ignore_errors=True)
                subprocess.run(["cp", "-al", src, tmp], check=True)
                open(os.path.join(tmp, "VAL"), "w").write(f"{u} {v}\n")
                shutil.rmtree(BEST, ignore_errors=True); os.rename(tmp, BEST)
                print(time.strftime("%H:%M"), f"best → step_{u:06d} (검증 {v})", flush=True)
    if not alive:
        print(time.strftime("%H:%M"), "트레이너 없음 — 끝", flush=True); break
    time.sleep(60)
