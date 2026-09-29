"""keep_best.py 시험 — 가짜 /proc 로 트레이너 판별(python train_b.py 만, tail·grep·좀비 제외), 가짜 run 폴더로 best 갱신·원본 삭제 뒤 생존."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keep_best as K


def fake_proc(entries):
    d = tempfile.mkdtemp()
    for pid, args in entries.items():
        os.makedirs(os.path.join(d, str(pid)))
        open(os.path.join(d, str(pid), "cmdline"), "wb").write(b"\0".join(a.encode() for a in args) + (b"\0" if args else b""))
    return d


def test_alive():
    assert not K.trainer_alive(proc=fake_proc({}))
    assert not K.trainer_alive(proc=fake_proc({10: ["tail", "-f", "/home/x/train_b.py.log"], 11: ["grep", "train_b.py", "a"], 12: ["vim", "train_b.py"], 13: []}))     # 13 = 좀비(빈 cmdline)
    p = fake_proc({10: ["tail", "-f", "train_b.py"], 20: ["/home/x/csm-venv/bin/python", "train_b.py", "--data", "d"]})
    assert K.trainer_alive(proc=p) and K.trainer_alive(pid=20, proc=p) and not K.trainer_alive(pid=10, proc=p) and not K.trainer_alive(pid=99, proc=p)
    assert K.trainer_alive(proc=fake_proc({30: ["python3", "/home/x/CSM/train_b.py", "--out", "r"]}))
    assert not K.trainer_alive(proc="/nonexistent/proc")
    print("  ✓ 트레이너 판별: python train_b.py 만, tail/grep/vim/좀비 제외, --pid")


def test_keep():
    run = tempfile.mkdtemp(); K.MIN_FREE = 0
    for u in (1500, 1750): os.makedirs(os.path.join(run, f"step_{u:06d}")); open(os.path.join(run, f"step_{u:06d}", "m.bin"), "w").write(str(u))
    open(os.path.join(run, "log.jsonl"), "w").write('{"update": 1500, "val_loss": 5.3659}\n{"update": 1600, "loss": 3.5}\n{"update": 1750, "val_loss": 5.3022}\n')
    assert K.keep_once(run) == (1750, 5.3022) and open(os.path.join(run, "best", "VAL")).read().split() == ["1750", "5.3022"]
    assert K.keep_once(run) is None                                                       # 같은 best 면 안 함
    open(os.path.join(run, "log.jsonl"), "a").write('{"update": 2000, "val_loss": 5.1}\n')
    assert K.keep_once(run) is None                                                       # step_002000 이 아직 없음
    os.makedirs(os.path.join(run, "step_002000")); open(os.path.join(run, "step_002000", "m.bin"), "w").write("2000")
    assert K.keep_once(run) == (2000, 5.1)
    import shutil; shutil.rmtree(os.path.join(run, "step_002000"))                        # 트레이너의 --keep 정리
    assert open(os.path.join(run, "best", "m.bin")).read() == "2000" and not os.path.exists(os.path.join(run, "best.tmp"))
    print("  ✓ best 갱신·중복 안 함·폴더 없으면 대기·원본 삭제 뒤 생존")


if __name__ == "__main__":
    test_alive(); test_keep(); print("keep_best 시험 2/2 통과")
