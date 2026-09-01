from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable


def loop_semaphore(value: int) -> Callable[[], asyncio.Semaphore]:
    """현재 이벤트 루프의 세마포어를 돌려주는 함수"""
    store: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
        weakref.WeakKeyDictionary()
    )

    def get() -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = store.get(loop)
        if semaphore is None:
            semaphore = store[loop] = asyncio.Semaphore(value)
        return semaphore

    return get
