import asyncio, logging
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from app.services.stt import GoogleSTTClient, AUDIO_EOS

async def main():
    client = GoogleSTTClient(
        project_id=os.environ.get("GOOGLE_PROJECT_ID"),
        location="us",
        model="chirp_3",
        language="ko-KR",
    )
    q: asyncio.Queue = asyncio.Queue()

    silence = b"\x00\x00" * 1600
    for _ in range(10):
        q.put_nowait(silence)
    q.put_nowait(AUDIO_EOS)

    print("STT 호출 시작", flush=True)
    try:
        async for ev in client.stream(q):
            print("EVENT:", ev, flush=True)
        print("스트림 정상 종료", flush=True)
    except Exception:
        logging.exception("STT 에러 발생")

asyncio.run(main())