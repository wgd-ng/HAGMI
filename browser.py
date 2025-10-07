import hashlib
import logging
from typing import Any, AsyncGenerator, TypedDict
import time
import re
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
from models.genai import GenerateContentRequest
from models import _adapter
from datetime import datetime, timezone
from patchright.async_api import Route, expect, async_playwright, BrowserContext, Browser, Page, TimeoutError
import dataclasses
from config import config, AIOHTTP_PROXY, AIOHTTP_PROXY_AUTH, CAMOUFOX_PROXY
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
            [[[[[None,"请求已转移到Python中。"]],"model"],1]],
            None,[6,9,215,None,[[1,6]],None,None,None,None,200]],
        [
            None,None,None,["1749019849541811",109021497,4162363067]]
    ]
]


@dataclasses.dataclass
class InterceptTask:
    request_id: str
    model: str
    request: GenerateContentRequest
    future: asyncio.Future
    profiler: Profiler


class AccountInfo(TypedDict):
    email: str | None
    apiKey: str | None


class BrowserPool:
    queue: asyncio.Queue[InterceptTask]
    _workers: dict[asyncio.Task, 'BrowserWorker']

    def __init__(self, credentialManager: CredentialManager, worker_count: int = config.WorkerCount, *, endpoint: str | None = None, loop=None):
        self._loop = loop
        self._endpoint = endpoint
        self.queue = asyncio.Queue()
        self._credMgr = credentialManager
        self._worker_count = min(worker_count, len(self._credMgr.credentials))
        self._workers = {}
        self._models: list[Model] = []

    def set_Models(self, models: list[Model]):
        self._models = models

    def get_Models(self) -> list[Model]:
        #TODO: model list override
        return self._models

    def worker_done_callback(self, task: asyncio.Task) -> None:
        if exc := task.exception():
            logger.error('worker %s failed with exception', task.get_name(), exc_info=exc)
        else:
            logger.info('worker %s exit normally', task.get_name())
        if worker := self._workers.pop(task, None):
            if not worker.ready.done():  # not ready 那就是验证登录状态的时候出错了
                worker.ready.set_result(None)
            elif exc:
                logging.fatal('worker %s unexpected exception', task.get_name(), exc_info=exc)

    async def start(self):

        for cred in self._credMgr.credentials[:self._worker_count]:

            worker = DirectBrowserWorker(
                credential=cred,
                pool=self,
                endpoint=self._endpoint,
                loop=self._loop
            )
            logger.info('spawn worker with %s', cred.email)
            task = asyncio.create_task(worker.run(), name=f"BrowserWorker-{cred.email}")
            task.add_done_callback(self.worker_done_callback)
            self._workers[task] = worker

            # sleep 10s。防止一次启动太多浏览器
            await asyncio.sleep(10)

        logger.info('waiting for workers to be ready')
        await asyncio.gather(*[
            worker.ready
            for task, worker in self._workers.items() if not task.done()
        ], return_exceptions=True)
        for task, worker in self._workers.items():
            account_info = await worker.ready
            if not account_info:
                continue
            if not worker._credential.email and account_info['email']:
                worker._credential.email = account_info['email']
            if not worker._credential.apikey and account_info['apiKey']:
                worker._credential.apikey = account_info['apiKey']

        if config.WorkerCount == 0:
            pass
        elif len(self._workers) <= 0:
            raise BaseException('No Worker Available')
        logger.info('%d Workers Up and Running', len(self._workers))
        return self

    async def stop(self):
        await asyncio.gather(*[
            worker.stop() for worker in self._workers.values()], return_exceptions=True)

    async def put_task(self, task: InterceptTask):
        await self.queue.put(task)


class BrowserWorker:
    _browser: BrowserContext | None
    _credential: Credential
    _endpoint: str | None
    _pool: BrowserPool
    status: str
    ready: asyncio.Future
    cookies: dict[str, str]

    def __init__(self, credential: Credential, pool: BrowserPool, *, endpoint: str|None = None, loop=None):
        self._loop = loop
        self._credential = credential
        self._browser = None
        self._endpoint = endpoint
        self._pages = []
        self._pool = pool
        self.ready = asyncio.Future()
        self.cookies = {}
    
    def _process_cookies(self, cookies: list[dict[str, str]]) -> dict[str, str]:
        return {
            item['name']: item['value']
            for item in cookies
        }

    async def browser(self) -> BrowserContext:
        if not self._browser:
            if not self._endpoint:
                # _browser = typing.cast(Browser, await AsyncCamoufox(
                #     headless=config.Headless,
                #     main_world_eval=True,
                #     enable_cache=True,
                #     locale="US",
                #     proxy=CAMOUFOX_PROXY,
                #     geoip=True if CAMOUFOX_PROXY else False,
                # ).__aenter__())
                playwright = await async_playwright().__aenter__()
                _browser = await playwright.chromium.launch(
                    headless=config.Headless is True,
                )
            else:
                _browser = await (await async_playwright().__aenter__()).chromium.connect(self._endpoint)
            if _browser.contexts:
                context = _browser.contexts[0]
            else:
                storage_state = None

                if self._credential.stateFile and os.path.exists(f'{config.StatesDir}/{self._credential.stateFile}'):
                    storage_state = f'{config.StatesDir}/{self._credential.stateFile}'
                    with open(storage_state) as fp:
                        self.cookies = self._process_cookies(json.load(fp)['cookies'])
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
            async with aiohttp.ClientSession() as session:

                resp = await session.post(
                    route.request.url,
                    headers=route.request.headers,
                    data=route.request.post_data,
                    proxy=AIOHTTP_PROXY, proxy_auth=AIOHTTP_PROXY_AUTH,
                    ssl=False if AIOHTTP_PROXY else True
                )

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
            await page.evaluate("""()=>{localStorage.setItem("aiStudioUserPreference", '{"isAdvancedOpen":false,"isSafetySettingsOpen":false,"areToolsOpen":true,"autosaveEnabled":false,"hasShownDrivePermissionDialog":true,"hasShownAutosaveOnDialog":true,"enterKeyBehavior":0,"theme":"system","bidiOutputFormat":3,"isSystemInstructionsOpen":true,"warmWelcomeDisplayed":true,"getCodeLanguage":"Python","getCodeHistoryToggle":true,"fileCopyrightAcknowledged":false,"enableSearchAsATool":true,"selectedSystemInstructionsConfigName":null,"thinkingBudgetsByModel":{},"rawModeEnabled":false,"monacoEditorTextWrap":false,"monacoEditorFontLigatures":true,"monacoEditorMinimap":false,"monacoEditorFolding":false,"monacoEditorLineNumbers":true,"monacoEditorStickyScrollEnabled":true,"monacoEditorGuidesIndentation":true}')}""")

    async def validate_state(self) -> AccountInfo | None:
        async with self.page() as page:
            logger.info('start validate state')
            await page.goto(f'{config.AIStudioUrl}/prompts/new_chat')
            if page.url.startswith(config.AIStudioUrl):
                return await self.fetch_account_info(page)
            elif not page.url.startswith('https://accounts.google.com/'):
                raise BaseException(f"Page at unexcpected URL: {page.url}")

            # 没登录
            if not self._credential.email or not self._credential.password:
                return None

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
            with open(f'{config.StatesDir}/{self._credential.stateFile}') as fp:
                self.cookies = self._process_cookies(json.load(fp)['cookies'])
            logger.info('store stete for credential %s', self._credential.email)
            return await self.fetch_account_info(page)

    async def fetch_account_info(self, page: Page) -> AccountInfo | None:
        rtn: AccountInfo = {
            'email': None,
            'apiKey': None
        }
        async def handle_apikeys(route: Route):
            await route.continue_()
            if not route.request.url.endswith('ListCloudApiKeys'):
                return

            if not (resp := await route.request.response()):
                return

            data = json.loads(await resp.body())

            rtn['apiKey'] = data[0][0][2]

        await page.route("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*", handle_apikeys)
        await page.goto('https://aistudio.google.com/api-keys')
        email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        await page.wait_for_load_state('load')
        global_data = await page.evaluate('window.WIZ_global_data', isolated_context=False)
        if not global_data:
            return None
        for _, value in global_data.items():
            if not isinstance(value, str):
                continue
            if email_regex.match(value):
                rtn['email'] = value
        await page.unroute("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*", handle_apikeys)
        return rtn

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

    async def stop(self):
        if self._browser:
            await self._browser.close()
            self._browser = None

    async def run(self):
        if not (accountInfo := await self.validate_state()):
            logger.info('State is not valid for credential %s', self._credential.email)
            await self.stop()
            return
        logger.info('Worker %s is ready', self._credential.email)
        self.ready.set_result(accountInfo)
        while True:
            task = await self._pool.queue.get()
            try:
                task.profiler.span('worker: task fetched')
                await self.InterceptRequest(task.request_id, task.model, task.request, task.future, task.profiler)
            except BaseException as exc:
                task.profiler.span('worker: failed with exception', traceback.format_exception(exc))
                task.future.set_exception(exc)

    async def InterceptRequest(self, request_id:str, model: str, request: GenerateContentRequest, future: asyncio.Future, profiler: Profiler, timeout: int=60):
        prompt_history = _adapter.GenAIRequestToAiStudioPromptHistory(model, request, prompt_id=request_id)
        profiler.span('adapter: GenAIRequestToAiStudioPromptHistory')
        prompt_id = request_id

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
                case 'GenerateAccessToken':
                    # 阻止保存Prompt至历史
                    await route.fulfill(
                        content_type='application/json+protobuf; charset=UTF-8',
                        body='[16,"Request is missing required authentication credential. Expected OAuth 2 access token, login cookie or other valid authentication credential. Seehttps://developers.google.com/identity/sign-in/web/devconsole-project."]',
                        status=401,
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

        try:
            async with asyncio.timeout(timeout):
                async with self.page() as page:
                    profiler.span('Page: Created')
                    await page.route("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*", handle_route)
                    await page.goto(f'{config.AIStudioUrl}/prompts/{prompt_id}')
                    profiler.span('Page: Loaded')
                    last_turn = page.locator('ms-chat-turn').last
                    await expect(last_turn.locator('ms-text-chunk')).to_have_text('(placeholder)', timeout=20000)
                    profiler.span('Page: Placeholder Visible')
                    # 到处点点
                    if await page.locator('.glue-cookie-notification-bar__reject').is_visible():
                        await page.locator('.glue-cookie-notification-bar__reject').click()
                    if await page.locator('button[aria-label="Close run settings panel"]').is_visible():
                        await page.locator('button[aria-label="Close run settings panel"]').click(force=True)
                    await page.locator('ms-text-chunk textarea').click()

                    while await last_turn.locator('ms-text-chunk').text_content() == '(placeholder)':
                        await last_turn.click(force=True)
                        profiler.span('Page: Last Turn Hover')
                        rerun = last_turn.locator('[name="rerun-button"]')
                        await expect(rerun).to_be_visible()
                        profiler.span('Page: Rerun Visible')
                        await last_turn.locator('[name="rerun-button"]').click()
                        profiler.span('Page: Rerun Clicked')
                        await asyncio.sleep(1)
                    await page.locator('ms-text-chunk textarea').click()
                    await future
                    await page.unroute("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/*")
        except:
            with open(f'{config.LogDir}/{prompt_id}.png', 'wb') as fp:
                fp.write(await page.screenshot())

            with open(f'{config.LogDir}/{prompt_id}.html', 'w') as fp:
                fp.write(await page.content())
            raise


POTOKEN_SCRIPT = '''(content) => {
    return new Promise(
        (resolve) => {
            window.__hagmi_snapshot(
                (resp) => {resolve(resp)},
                [{"content": content},undefined,undefined,undefined]
            )})}
'''


class DirectBrowserWorker(BrowserWorker):

    async def prepare_page(self, page: Page) -> None:
        await page.route("**/www.google-analytics.com/*", lambda route: route.abort())
        await page.route("**/play.google.com/*", lambda route: route.abort())
        await page.route("**/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/ListModels", self.handle_ListModels)

        regex = re.compile(r'\w\.resolve\(\{\w+:(\w+),\w+:\w+,\w+:\w+,\w+:\w+\}\)')

        async def modify_script(route: Route):
            if not route.request.url.endswith('m=_b'):
                return await route.fallback()
            resp = await route.fetch()
            script = await resp.text()
            if not (m := regex.search(script)):
                return await route.fallback()
            begin, _ = m.span()
            func_name = m.group(1)
            script = script[:begin] + (f'console.log("funccc", {func_name});window.__hagmi_snapshot={func_name};') + script[begin:]

            await route.fulfill(
                headers=resp.headers,
                body=script,
            )

        await page.route("**/boq-makersuite/_/js/**", modify_script)

        if not page.url.startswith(config.AIStudioUrl):
            await page.goto(f'{config.AIStudioUrl}/prompts/new_chat')
            await page.evaluate("""()=>{localStorage.setItem("aiStudioUserPreference", '{"isAdvancedOpen":false,"isSafetySettingsOpen":false,"areToolsOpen":true,"autosaveEnabled":false,"hasShownDrivePermissionDialog":true,"hasShownAutosaveOnDialog":true,"enterKeyBehavior":0,"theme":"system","bidiOutputFormat":3,"isSystemInstructionsOpen":true,"warmWelcomeDisplayed":true,"getCodeLanguage":"Python","getCodeHistoryToggle":true,"fileCopyrightAcknowledged":false,"enableSearchAsATool":true,"selectedSystemInstructionsConfigName":null,"thinkingBudgetsByModel":{},"rawModeEnabled":false,"monacoEditorTextWrap":false,"monacoEditorFontLigatures":true,"monacoEditorMinimap":false,"monacoEditorFolding":false,"monacoEditorLineNumbers":true,"monacoEditorStickyScrollEnabled":true,"monacoEditorGuidesIndentation":true}')}""")

    async def InterceptRequest(self, request_id:str, model: str, request: GenerateContentRequest, future: asyncio.Future, profiler: Profiler, timeout: int=60):
        req = _adapter.GenAIRequestToAiStudioRequest(f'models/{model}', request)
        text = []
        for content in request.contents:
            for part in content.parts:
                if part.text:
                    text.append(part.text)
        plaintext = ' '.join(text).encode()
        checksum = hashlib.sha256(plaintext).hexdigest()
        headers = {
            'accept': '*/*',
            'accept-language': 'US',
            'authorization': sapisidhash(self.cookies),
            'content-type': 'application/json+protobuf',
            'origin': 'https://aistudio.google.com',
            # 'priority': 'u=1, i',
            'referer': 'https://aistudio.google.com/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'x-goog-api-key': 'AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs',
            'x-goog-authuser': '0',
            'x-goog-ext-519733851-bin': 'CAASAUQwATgEQAA=',
            'x-user-agent': 'grpc-web-javascript/0.1',
            'cookie': '; '.join([f'{key}={value}' for key, value in self.cookies.items()]),
        }
        async with self.page() as page:
            # 选择随机元素移动鼠标
            spans = await page.locator("span:visible").all()
            random.shuffle(spans)
            for span in spans[:3]:
                await span.hover()
            req.potoken = await page.evaluate(POTOKEN_SCRIPT, checksum, isolated_context=False)
        body = flatten(req)
        future.set_result((headers, json.dumps(body, separators=(',', ':'))))
 