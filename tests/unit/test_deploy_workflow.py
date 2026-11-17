import subprocess
import textwrap
from pathlib import Path

DEPLOY_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "deploy.yml"


def _ec2_script() -> str:
    lines = DEPLOY_YML.read_text().splitlines()
    step = next(
        (i for i, line in enumerate(lines) if line.strip().removeprefix("- ") == "name: Deploy on EC2"),
        None,
    )
    if step is None:
        raise AssertionError(f"'Deploy on EC2' 스텝을 찾지 못했다: {DEPLOY_YML}")

    for idx in range(step, len(lines)):
        if lines[idx].strip() != "script: |":
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        body: list[str] = []
        for line in lines[idx + 1 :]:
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
        return textwrap.dedent("\n".join(body))

    raise AssertionError(f"'Deploy on EC2' 뒤에 'script: |' 블록이 없다: {DEPLOY_YML}")


def _prune_command() -> str:
    for line in _ec2_script().splitlines():
        if "docker image prune" in line:
            return line.strip()
    raise AssertionError("docker image prune 호출이 사라졌다 — 배포 이미지가 무한 누적된다")


def _prune_flags() -> set[str]:
    flags: set[str] = set()
    for token in _prune_command().split():
        if token.startswith("--"):
            flags.add(token)
        elif token.startswith("-"):
            flags.update(token[1:])
    return flags


def _job(name: str) -> list[str]:
    lines = DEPLOY_YML.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line == f"  {name}:"), None)
    if start is None:
        raise AssertionError(f"'{name}' 잡을 찾지 못했다: {DEPLOY_YML}")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= 2:
            break
        body.append(line)
    return body


def _needs(name: str) -> set[str]:
    for line in _job(name):
        if line.startswith("    needs:"):
            value = line.split(":", 1)[1].strip(" []")
            return {item.strip() for item in value.split(",") if item.strip()}
    return set()


def test_ec2_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n"], input=_ec2_script(), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_prune_reclaims_sha_tagged_images() -> None:
    flags = _prune_flags()
    assert "a" in flags or "--all" in flags, (
        f"배포 이미지는 커밋 SHA 태그가 붙어 dangling 이 되지 않는다 — "
        f"-a 없이는 한 개도 회수하지 못한다: {_prune_command()}"
    )


def test_prune_keeps_recent_images_for_rollback() -> None:
    assert "until=" in _prune_command(), (
        f"나이 필터가 없으면 롤백용 최근 이미지까지 지워진다: {_prune_command()}"
    )


def test_prune_failure_cannot_fail_a_successful_deploy() -> None:
    assert "|| true" in _prune_command(), (
        f"set -e 아래라 정리 실패가 배포 성공을 CD 실패로 뒤집는다: {_prune_command()}"
    )


def test_deploy_waits_for_tests_and_image() -> None:
    assert _needs("deploy") == {"test", "build-push"}, (
        "빌드가 시험과 나란히 돌아서 deploy 가 둘 다 기다려야 한다 — "
        f"test 가 빠지면 시험 안 된 이미지가 배포된다: {sorted(_needs('deploy'))}"
    )


def test_image_build_does_not_wait_for_tests() -> None:
    assert "test" not in _needs("build-push"), "build-push 가 test 를 기다리면 CD 가 다시 직렬이 된다"


def test_build_pushes_only_the_commit_tag() -> None:
    tagged = [line.strip() for line in _job("build-push") if "latest" in line.replace("ubuntu-latest", "")]
    assert not tagged, (
        "빌드는 시험이 끝나기 전에 올라간다 — 여기서 latest 를 붙이면 시험이 깨진 커밋 이미지에도 붙는다: "
        f"{tagged}"
    )
