from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_VERSION = "3.05"
_MODE_CONSOLE = "Console (in: all wav-files in input path, out: one textfile)"
_AMP_ENVELOPE = "Envelope [To AmplitudeTier (period)]"
_UNDEFINED_OUT = "indeterminate values are -- undefined --"

_RESULT_REL = Path("results") / "tremor_resCon.txt"
_SOUNDS_REL = "./sounds/"

_INTERCEPT = 2.445
_COEF = {"ftrcip": 0.467, "atri": 0.077, "fcohnr": 0.102, "fcom": -1.405}

_WANTED = {"FTrCIP": "ftrcip", "ATrI": "atri", "FCoHNR": "fcohnr", "FCoM": "fcom"}

_UNDEFINED = {"", "--undefined--", "undefined", "nan", "?"}


class AvtiStatus(StrEnum):
    OK = "OK"
    NO_SUSTAINED = "NO_SUSTAINED"   # 3초 지속발성 구간이 없음 (대화에서는 이게 기본)
    UNDEFINED = "UNDEFINED"         # 스크립트가 값을 못 냄
    TIMEOUT = "TIMEOUT"             # Praat 응답 없음
    NO_SCRIPT = "NO_SCRIPT"         # praat 바이너리/스크립트 미설치
    ERROR = "ERROR"


@dataclass
class AvtiPart:
    """파트 하나의 측정 결과. 실패해도 행은 남긴다."""

    part_index: int          # 0 = 세션 전체, 1..n = 사용자 턴
    start_sec: float         # 실제로 분석에 쓴 구간 (실패 시 파트 구간)
    end_sec: float
    status: AvtiStatus
    avti: float | None = None
    ftrcip: float | None = None
    atri: float | None = None
    fcohnr: float | None = None
    fcom: float | None = None


@dataclass
class AvtiConfig:
    sample_rate: int = 16000
    window_sec: float = 3.0
    min_sustained_sec: float = 3.0
    timeout_sec: float = 30.0

    analysis_time_step: float = 0.015
    min_pitch: float = 60
    max_pitch: float = 350
    silence_threshold: float = 0.03
    voicing_threshold: float = 0.3
    octave_cost: float = 0.01
    octave_jump_cost: float = 0.35
    voiced_unvoiced_cost: float = 0.14
    min_tremor_hz: float = 1.5
    max_tremor_hz: float = 15
    contour_magnitude_threshold: float = 0.01
    tremor_cyclicality_threshold: float = 0.15
    freq_tremor_octave_cost: float = 0.01
    amp_tremor_octave_cost: float = 0.01

    def form_args(self) -> list[str]:
        """스크립트 form 순서대로. 하나라도 어긋나면 조용히 엉뚱한 값이 나온다."""
        return [
            _MODE_CONSOLE,
            _SOUNDS_REL,
            str(self.analysis_time_step),
            str(self.min_pitch),
            str(self.max_pitch),
            str(self.silence_threshold),
            str(self.voicing_threshold),
            str(self.octave_cost),
            str(self.octave_jump_cost),
            str(self.voiced_unvoiced_cost),
            _AMP_ENVELOPE,
            str(self.min_tremor_hz),
            str(self.max_tremor_hz),
            str(self.contour_magnitude_threshold),
            str(self.tremor_cyclicality_threshold),
            str(self.freq_tremor_octave_cost),
            str(self.amp_tremor_octave_cost),
            _UNDEFINED_OUT,
        ]


def compute_avti(*, ftrcip: float, atri: float, fcohnr: float, fcom: float) -> float:
    return (
        _INTERCEPT
        + _COEF["ftrcip"] * ftrcip
        + _COEF["atri"] * atri
        + _COEF["fcohnr"] * fcohnr
        + _COEF["fcom"] * fcom
    )


def _to_float(raw: str) -> float | None:
    """'undefined' 계열은 None. 0으로 바꾸지 않는다."""
    token = raw.strip()
    if token.lower() in _UNDEFINED:
        return None
    try:
        return float(token)
    except ValueError:
        return None


@lru_cache(maxsize=8)
def supports_full_trust(praat_bin: str) -> bool:
    """Praat 7 부터 스크립트의 파일 삭제에 --FULL-TRUST 가 필요하다."""
    try:
        proc = subprocess.run(
            [praat_bin, "--help"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "FULL-TRUST" in (proc.stdout + proc.stderr)


def _normalize_column(column: str) -> str:
    return re.sub(r"\[.*?\]", "", column).strip()


def parse_tremor_table(text: str) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    header: list[str] | None = None
    name_col = 0

    for line in text.splitlines():
        if not line.strip():
            continue
        cells = _split_row(line)
        if header is None:
            columns = [_normalize_column(c) for c in cells]
            if not any(c in _WANTED for c in columns):
                continue  # 헤더 전 로그/배너
            header = columns
            name_col = next(
                (
                    i
                    for i, c in enumerate(header)
                    if c.lower() in ("soundname", "file", "filename", "name", "sound")
                ),
                0,
            )
            continue

        if len(cells) < len(header):
            continue
        name = Path(cells[name_col].strip()).stem
        values: dict[str, float | None] = {}
        for i, column in enumerate(header):
            key = _WANTED.get(column)
            if key is not None:
                values[key] = _to_float(cells[i])
        if values:
            rows[name] = values
    return rows


def _split_row(line: str) -> list[str]:
    for sep in ("\t", ","):
        if sep in line:
            return line.split(sep)
    return line.split()


class AvtiAnalyzer:
    """지속발성 구간을 잘라 Praat 에 한 번에 넘기고 결과를 표로 받는다."""

    def __init__(
        self,
        *,
        praat_bin: str,
        script_path: str | None,
        config: AvtiConfig | None = None,
    ) -> None:
        self._praat = praat_bin
        self._script = Path(script_path).expanduser() if script_path else None
        self.config = config or AvtiConfig()

    @property
    def available(self) -> bool:
        return (
            self._script is not None
            and self._script.is_file()
            and (self._script.parent / "procedures").is_dir()
            and shutil.which(self._praat) is not None
        )

    def pick_window(
        self, part: tuple[float, float], sustained_spans: list
    ) -> tuple[float, float] | None:
        """파트 안에서 가장 긴 지속발성 런의 가운데 window_sec 만큼."""
        # 창이 런보다 길면 지속발성 밖까지 잘라내게 된다. 둘 중 큰 쪽을 요구한다.
        need = max(self.config.min_sustained_sec, self.config.window_sec)
        best: tuple[float, float] | None = None
        best_len = 0.0
        for span_start, span_end in sustained_spans:
            lo = max(part[0], span_start)
            hi = min(part[1], span_end)
            length = hi - lo
            if length >= need and length > best_len:
                best, best_len = (lo, hi), length
        if best is None:
            return None

        # 시작/끝의 상승·하강을 피해 가운데를 쓴다 (논문: 앞뒤 무음 없는 구간 선택).
        center = (best[0] + best[1]) / 2
        half = self.config.window_sec / 2
        return (round(center - half, 3), round(center + half, 3))

    def analyze(
        self, pcm: bytes, parts: list[tuple[float, float]], sustained_spans: list
    ) -> list[AvtiPart]:
        """parts[0] 은 세션 전체, 그 뒤는 사용자 턴 순서."""
        results: list[AvtiPart] = []
        windows: dict[str, int] = {}  # wav stem → part_index

        if not self.available:
            return [
                AvtiPart(i, round(p[0], 3), round(p[1], 3), AvtiStatus.NO_SCRIPT)
                for i, p in enumerate(parts)
            ]

        with tempfile.TemporaryDirectory(prefix="avti-") as tmp:
            workdir = self._prepare_workdir(Path(tmp))
            tmpdir = workdir / "sounds"
            for index, part in enumerate(parts):
                window = self.pick_window(part, sustained_spans)
                if window is None:
                    results.append(
                        AvtiPart(index, round(part[0], 3), round(part[1], 3),
                                 AvtiStatus.NO_SUSTAINED)
                    )
                    continue
                stem = f"part{index:03d}"
                try:
                    self._write_wav(tmpdir / f"{stem}.wav", pcm, window)
                except Exception:
                    logger.warning("AVTI 구간 추출 실패", exc_info=True,
                                   extra={"part_index": index})
                    results.append(AvtiPart(index, window[0], window[1], AvtiStatus.ERROR))
                    continue
                windows[stem] = index
                results.append(AvtiPart(index, window[0], window[1], AvtiStatus.ERROR))

            if not windows:
                return sorted(results, key=lambda r: r.part_index)

            try:
                table = self._run_praat(tmpdir)
            except subprocess.TimeoutExpired:
                logger.warning("AVTI Praat 타임아웃", extra={"parts": len(windows)})
                self._mark(results, windows.values(), AvtiStatus.TIMEOUT)
                return sorted(results, key=lambda r: r.part_index)
            except Exception:
                logger.warning("AVTI Praat 실행 실패", exc_info=True)
                self._mark(results, windows.values(), AvtiStatus.ERROR)
                return sorted(results, key=lambda r: r.part_index)

        by_index = {r.part_index: r for r in results}
        for stem, index in windows.items():
            row = table.get(stem)
            target = by_index[index]
            if row is None:
                target.status = AvtiStatus.UNDEFINED
                continue
            values = {key: row.get(key) for key in _COEF}
            if any(v is None for v in values.values()):
                # 넷 중 하나라도 못 재면 AVTI 는 없다. 잰 값은 그대로 남긴다.
                target.status = AvtiStatus.UNDEFINED
                target.ftrcip, target.atri = values["ftrcip"], values["atri"]
                target.fcohnr, target.fcom = values["fcohnr"], values["fcom"]
                continue
            target.status = AvtiStatus.OK
            target.ftrcip, target.atri = values["ftrcip"], values["atri"]
            target.fcohnr, target.fcom = values["fcohnr"], values["fcom"]
            target.avti = round(compute_avti(**values), 4)

        return sorted(results, key=lambda r: r.part_index)

    @staticmethod
    def _mark(results: list[AvtiPart], indices, status: AvtiStatus) -> None:
        wanted = set(indices)
        for result in results:
            if result.part_index in wanted:
                result.status = status

    def _write_wav(self, path: Path, pcm: bytes, window: tuple[float, float]) -> None:
        sample_rate = self.config.sample_rate
        start = max(0, int(window[0] * sample_rate)) * 2
        end = min(len(pcm), int(window[1] * sample_rate) * 2)
        chunk = pcm[start:end]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            raise ValueError("빈 구간")
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes(chunk)

    def _prepare_workdir(self, root: Path) -> Path:
        """스크립트 트리를 세션 전용 디렉터리로 복사한다."""

        assert self._script is not None
        source = self._script.parent
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._script, root / self._script.name)
        shutil.copytree(source / "procedures", root / "procedures")
        (root / "sounds").mkdir()
        (root / "results").mkdir()
        return root

    def _run_praat(self, wav_dir: Path) -> dict[str, dict[str, float | None]]:
        """폴더 일괄 분석 모드로 한 번만 호출한다."""
        workdir = wav_dir.parent
        script = workdir / Path(self._script).name
        trust = ["--FULL-TRUST"] if supports_full_trust(self._praat) else []
        argv = [
            self._praat, "--run", *trust, str(script),
            *self.config.form_args(),
        ]
        proc = subprocess.run(
            argv,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        out_path = workdir / _RESULT_REL
        if not out_path.is_file():
            detail = (proc.stderr or proc.stdout).strip()[:300]
            raise RuntimeError(f"praat exit={proc.returncode}, 결과 파일 없음: {detail}")
        return parse_tremor_table(out_path.read_text(encoding="utf-8", errors="replace"))
