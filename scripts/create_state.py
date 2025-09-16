import sys
import logging
sys.path.insert(0, '.')
import os
import re
import asyncio
import concurrent.futures
import argparse

import aiohttp
from camoufox.async_api import AsyncCamoufox
from playwright.async_api import expect

from config import config


parser = argparse.ArgumentParser(
    description="登录AI Studio 并保存登录状态",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--email', type=str, help='Google 帐号自动填充', metavar='user@example.com',
)

parser.add_argument(
    '--password', type=str, help='Google 帐号密码自动填充(可选)'
)

parser.add_argument(
    '--remote', type=str, help='推送到远程HAGMI实例', metavar='http://127.0.0.1:8000'
)

args = parser.parse_args()

async def login():
    proxy = None
    if config.Proxy:
        proxy = {
            "server": config.Proxy.server,
            "username": config.Proxy.username,
            "password": config.Proxy.password,
        }
    storage_state = None
    if os.path.exists(f'{config.StatesDir}/{args.email}.json'):
        storage_state = f'{config.StatesDir}/{args.email}.json'
    async with AsyncCamoufox(
                main_world_eval=True,
                headless=False,
                proxy=proxy,
                geoip=True if proxy else False,
            ) as browser:
        context = await browser.new_context(storage_state=storage_state)
        page = await context.new_page()
        await page.goto(f'{config.AIStudioUrl}/prompts/new_chat')
        if page.url.startswith('https://accounts.google.com/'):
            # 跳转到登录页面了
            if args.email:
                await page.locator('input#identifierId').type(args.email)
                await expect(page.locator('#identifierNext button')).to_be_enabled()
                await page.locator('#identifierNext button').click()
                if args.password:
                    await asyncio.sleep(3)
                    await expect(page.locator('input[name="Passwd"]')).to_be_editable()
                    await page.locator('input[name="Passwd"]').type(args.password)
                    await expect(page.locator('#passwordNext button')).to_be_enabled()
                    await page.locator('#passwordNext button').click()

        # 等待登录成功跳转到AI Studio
        await page.wait_for_url(f'{config.AIStudioUrl}/prompts/new_chat', timeout=0)
        # 尝试从页面数据中获取帐号
        regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        emails = []
        for _ in range(15):
            global_data = await page.evaluate('mw:window.WIZ_global_data')
            if not global_data:
                await asyncio.sleep(1)
                continue
            emails = [v for v in (global_data).values() if isinstance(v, str) and regex.match(v)]
        auto_email = emails[0] if len(emails) == 1 else None
        email = args.email
        if not email and auto_email:
            email = auto_email
        loop = asyncio.get_running_loop()
        while not email:
            email = await loop.run_in_executor(
                concurrent.futures.ThreadPoolExecutor(),
                input,
                '无法自动获取帐号, 请输入帐号以保存登录状态:'
            )

        state_path = f'{config.StatesDir}/{email}.json'
        await context.storage_state(path=state_path)
        if args.remote:
            async with aiohttp.ClientSession() as session:
                url = f'{args.remote}/admin/upload_state'
                with open(state_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        'state',
                        f,
                        filename=f'{email}.json',
                        content_type='application/octet-stream'
                    )
                    async with session.post(url, params={'key': config.AuthKey}, data=data) as resp:
                        print(f"Status: {resp.status}")
                        print(f"Response: {await resp.text()}")


asyncio.run(login())