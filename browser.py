import hashlib
import logging
from typing import Any, AsyncGenerator
import time
import os
import typing
import json
import asyncio
import traceback
import random
import string
import aiohttp
import contextlib
import pathlib
from models.aistudio import PromptHistory, flatten, inflate, StreamParser, Model, ListModelsResponse
from datetime import datetime, timezone
from camoufox.async_api import AsyncCamoufox, AsyncNewBrowser
from playwright.async_api import Route, expect, async_playwright, BrowserContext, Browser, Page
import dataclasses
from config import config
from utils import Profiler, CredentialManager, Credential


logger = logging.getLogger('Browser')


def sapisidhash(cookies: dict[str, str]) -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    sapisid: str = cookies.get(
        '__Secure-3PAPISID',
        cookies.get('SAPISID', cookies.get('__Secure-1PAPISID'))) or ''
    assert sapisid
    m = hashlib.sha1(' '.join(
        (str(now), sapisid, "https://aistudio.google.com")).encode())
    sapisidhash = '_'.join((str(now), m.hexdigest()))
    return ' '.join(
        f'{key} {sapisidhash}' for key in ('SAPISIDHASH', 'SAPISID1PHASH', 'SAPISID3PHASH'))



fixed_responsed = [
    [
        [
            [[[[[None,"**Thinking**\n\n不，你不想。\n\n\n",None,None,None,None,None,None,None,None,None,None,1]],"model"]]],
            None,[6,None,74,None,[[1,6]],None,None,None,None,68]],
        [
            [[[[[None,"摆。"]],"model"],1]],
            None,[6,9,215,None,[[1,6]],None,None,None,None,200]],
        [
            None,None,None,["1749019849541811",109021497,4162363067]]
    ]
]


@dataclasses.dataclass
class InterceptTask:
    prompt_history: PromptHistory
    future: asyncio.Future
    profiler: Profiler


class BrowserPool:
    queue: asyncio.Queue[InterceptTask]

    def __init__(self, credentialManager: CredentialManager, worker_count: int = config.WorkerCount, *, endpoint: str | None = None, loop=None):
        self._loop = loop
        self._endpoint = endpoint
        self.queue = asyncio.Queue()
        self._credMgr = credentialManager
        self._worker_count = min(worker_count, len(self._credMgr.credentials))
        self._workers: list[asyncio.Task] = []
        self._models: list[Model] = []

    def set_Models(self, models: list[Model]):
        self._models = models

    def get_Models(self) -> list[Model]:
        #TODO: model list override
        return self._models

    async def start(self):

        for cred in self._credMgr.credentials[:self._worker_count]:

            worker = BrowserWorker(
                credential=cred,
                pool=self,
                endpoint=self._endpoint,
                loop=self._loop
            )
            logger.info('spawn worker with %s', cred.email)
            task = asyncio.create_task(worker.run(), name=f"BrowserWorker-{cred.email}")
            self._workers.append(task)
        return self

    async def stop(self):
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def put_task(self, task: InterceptTask):
        await self.queue.put(task)


class BrowserWorker:
    _browser: BrowserContext | None
    _credential: Credential
    _endpoint: str | None
    _pool: BrowserPool
    status: str

    def __init__(self, credential: Credential, pool: BrowserPool, *, endpoint: str|None = None, loop=None):
        self._loop = loop
        self._credential = credential
        self._browser = None
        self._endpoint = endpoint
        self._pages = []
        self._pool = pool

    async def browser(self) -> BrowserContext:
        if not self._browser:
            if not self._endpoint:
                _browser = typing.cast(Browser, await AsyncCamoufox(
                    headless=config.Headless,
                    main_world_eval=True,
                    enable_cache=True,
                    locale="US",
                ).__aenter__())
            else:
                _browser = await (await async_playwright().__aenter__()).firefox.connect(self._endpoint)
            if _browser.contexts:
                context = _browser.contexts[0]
            else:
                storage_state = None

                if self._credential.stateFile and os.path.exists(f'{config.StatesDir}/{self._credential.stateFile}'):
                    storage_state = f'{config.StatesDir}/{self._credential.stateFile}'
                context = await _browser.new_context(
                    storage_state=storage_state,
                    ignore_https_errors=True,
                    locale="US",
                )
            self._browser = context
            self._pages = list(context.pages)
        return self._browser

    async def handle_ListModels(self, route: Route) -> None:
        if not self._pool.get_Models():
            resp = await route.fetch()
            data = inflate(await resp.json(), ListModelsResponse)
            if data:
                self._pool.set_Models(data.models)
        # TODO: 从缓存快速返回请求
        # TODO: 劫持并修改模型列表
        await route.fallback()

    async def prepare_page(self, page: Page) -> None:
        await page.route("**/www.google-analytics.com/*", lambda route: route.abort())
        await page.route("**/play.google.com/*", lambda route: route.abort())
        await page.route("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/ListModels", self.handle_ListModels)
        if not page.url.startswith(config.AIStudioUrl):
            await page.goto(f'{config.AIStudioUrl}/prompts/new_chat')
            await page.evaluate("""()=>{mw:localStorage.setItem("aiStudioUserPreference", '{"isAdvancedOpen":false,"isSafetySettingsOpen":false,"areToolsOpen":true,"autosaveEnabled":false,"hasShownDrivePermissionDialog":true,"hasShownAutosaveOnDialog":true,"enterKeyBehavior":0,"theme":"system","bidiOutputFormat":3,"isSystemInstructionsOpen":true,"warmWelcomeDisplayed":true,"getCodeLanguage":"Python","getCodeHistoryToggle":true,"fileCopyrightAcknowledged":false,"enableSearchAsATool":true,"selectedSystemInstructionsConfigName":null,"thinkingBudgetsByModel":{},"rawModeEnabled":false,"monacoEditorTextWrap":false,"monacoEditorFontLigatures":true,"monacoEditorMinimap":false,"monacoEditorFolding":false,"monacoEditorLineNumbers":true,"monacoEditorStickyScrollEnabled":true,"monacoEditorGuidesIndentation":true}')}""")

    async def validate_state(self) -> bool:
        async with self.page() as page:
            logger.info('start validate state')
            await page.goto(f'{config.AIStudioUrl}/prompts/new_chat')
            if page.url.startswith(config.AIStudioUrl):
                return True
            elif not page.url.startswith('https://accounts.google.com/'):
                raise BaseException(f"Page at unexcpected URL: {page.url}")

            # 没登录
            if not self._credential.email or not self._credential.password:
                return False

            logger.info('login using credential %s', self._credential.email)
            await page.locator('input#identifierId').type(self._credential.email)
            await expect(page.locator('#identifierNext button')).to_be_enabled()
            await page.locator('#identifierNext button').click()
            await asyncio.sleep(3)
            await expect(page.locator('input[name="Passwd"]')).to_be_editable()
            logger.info('login using credential %s type in password', self._credential.email)
            await page.locator('input[name="Passwd"]').type(self._credential.password)
            await expect(page.locator('#passwordNext button')).to_be_enabled()
            await page.locator('#passwordNext button').click()
            await page.wait_for_url(f'{config.AIStudioUrl}/prompts/new_chat')
            if await page.locator('mat-dialog-content .welcome-option button[aria-label="Try Gemini"]').count() > 0:
                await page.locator('mat-dialog-content .welcome-option button[aria-label="Try Gemini"]').click()
            await (await self.browser()).storage_state(path=f'{config.StatesDir}/{self._credential.stateFile}')
            logger.info('store stete for credential %s', self._credential.email)
            return True

    @contextlib.asynccontextmanager
    async def page(self):
        if not self._pages:
            page = await (await self.browser()).new_page()
        else:
            page = self._pages.pop(0)
        await self.prepare_page(page)
        try:
            yield page
        finally:
            self._pages.append(page)
            await page.unroute_all()

    async def run(self):
        await self.validate_state()
        logger.info('Worker %s is ready', self._credential.email)
        while True:
            try:
                task = await self._pool.queue.get()
                task.profiler.span('worker: task fetched')
                await self.InterceptRequest(task.prompt_history, task.future, task.profiler)
            except Exception as exc:
                task.profiler.span('worker: failed with exception', traceback.format_exception(exc))
                task.future.set_exception(exc)

    async def InterceptRequest(self, prompt_history: PromptHistory, future: asyncio.Future, profiler: Profiler, timeout: int=60):
        prompt_id = prompt_history.prompt.uri.split('/')[-1]

        async def handle_route(route: Route) -> None:
            match route.request.url.split('/')[-1]:
                case 'GenerateContent':
                    profiler.span('Route: Intercept GenerateContent')
                    future.set_result((route.request.headers, route.request.post_data))
                    await route.fulfill(
                        content_type='application/json+protobuf; charset=UTF-8',
                        body=json.dumps(fixed_responsed, separators=(',', ':'))
                    )
                case 'ResolveDriveResource':
                    profiler.span('Route: serve PromptHistory')
                    data = json.dumps(flatten(prompt_history), separators=(',', ':'))
                    await route.fulfill(
                        content_type='application/json+protobuf; charset=UTF-8',
                        body=data,
                    )
                case 'CreatePrompt':
                    await route.abort()
                case 'UpdatePrompt':
                    await route.abort()
                case 'ListPrompts':
                    profiler.span('Route: serve ListPrompts')
                    data = json.dumps(flatten([prompt_history.prompt.promptMetadata]), separators=(',', ':'))
                    await route.fulfill(
                        content_type='application/json+protobuf; charset=UTF-8',
                        body=data,
                    )
                case 'CountTokens':
                    await route.fulfill(
                        content_type='application/json+protobuf; charset=UTF-8',
                        body=json.dumps([4,[],[[[3],1]],None,None,[[1,4]]],  separators=(',', ':'))
                    )
                case _:
                    await route.fallback()

        async with asyncio.timeout(timeout):
            async with self.page() as page:
                profiler.span('Page: Created')
                await page.route("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*", handle_route)
                await page.goto(f'{config.AIStudioUrl}/prompts/{prompt_id}')
                profiler.span('Page: Loaded')
                last_turn = page.locator('ms-chat-turn').last
                await expect(last_turn.locator('ms-text-chunk')).to_have_text('(placeholder)', timeout=20000)
                profiler.span('Page: Placeholder Visible')
                if await page.locator('.glue-cookie-notification-bar__reject').is_visible():
                    await page.locator('.glue-cookie-notification-bar__reject').click()
                if await page.locator('button[aria-label="Close run settings panel"]').is_visible():
                    await page.locator('button[aria-label="Close run settings panel"]').click(force=True)
                await page.locator('ms-text-chunk textarea').click()
                await last_turn.click(force=True)
                profiler.span('Page: Last Turn Hover')
                rerun = last_turn.locator('[name="rerun-button"]')
                await expect(rerun).to_be_visible()
                profiler.span('Page: Rerun Visible')
                await rerun.click(force=True)
                profiler.span('Page: Rerun Clicked')
                await page.locator('ms-text-chunk textarea').click()
                await future
                await page.unroute("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*")
