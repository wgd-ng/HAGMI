from typing import AsyncGenerator, Any
import logging
import json
import time
import os
import traceback
import config
import random
from config import config, Credential

logger = logging.getLogger('TinyProfiler')
LOGDIR = config.LogDir
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)


async def TinyProfiler(topic: str | None=None) -> AsyncGenerator[None, str | None]:
    start = time.time()
    last = start
    while True:
        text = yield
        if text is None:
            continue
        cur = time.time()
        diff, last = cur - last, cur
        logger.debug('%s: %02.2f(+ %02.2f) Reach %s', topic, cur - start, diff, text)


class Profiler:
    history: list[tuple[float, str, Any | None]]

    def __init__(self, topic: str | None = None):
        self.topic = topic
        self.last_time = self.start_time = time.time()
        self.history = []

    def __enter__(self):
        return self

    def save(self):
        logger.warning('Profiler[%s] Save Log', self.topic)
        with open(f'{LOGDIR}/{self.topic}.json', 'w') as fp:
            json.dump({'history': self.history}, fp, separators=(',', ': '))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (not exc_type or not self.topic) and not config.Debug:
            return
        logger.warning('Profiler[%s] Save Log With Error: %s', self.topic, exc_val)
        with open(f'{LOGDIR}/{self.topic}.json', 'w') as fp:
            json.dump({'history': self.history, 'traceback': traceback.format_exception(exc_type, exc_val, exc_tb)}, fp, separators=(',', ': '))

    def span(self, text: str, extra: Any | None = None):
        current_time = time.time()
        diff = current_time - self.last_time
        self.last_time = current_time
        total_elapsed = current_time - self.start_time
        s = f'{self.topic or "Profiler"}: {total_elapsed:.2f}({+diff:.2f}) Reach {text}'
        logger.debug(s)
        self.history.append((diff, s, extra))


class CredentialManager():
    credentials: list[Credential]
    idx: int

    def __init__(self, credentials: list[Credential]) -> None:
        self.credentials = list(credentials)
        random.shuffle(self.credentials)
        self.idx = 0

    @property
    def api_key(self) -> str:
        '''轮询可用的API KEY'''
        for _ in self.credentials:
            idx = self.idx
            self.idx = (self.idx + 1) % len(self.credentials)
            credential = self.credentials[idx]
            if credential.apikey is not None:
                return credential.apikey
        raise ValueError('No API key available')
