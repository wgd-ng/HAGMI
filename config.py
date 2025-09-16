import os
import logging
from typing import Literal
import aiohttp
from pydantic import BaseModel, config, Field
import yaml


class Credential(BaseModel):
    email: str | None = None
    password: str | None = None
    apikey: str | None = None
    stateFile: str | None = None


class ProxyConfig(BaseModel):
    server: str
    username: str | None = None
    password: str | None = None


class Config(BaseModel):
    Debug: bool = False
    LogDir: str = './gemini_logs'
    StatesDir: str = './states'
    AIStudioAPIUrl: str = 'https://alkalimakersuite-pa.clients6.google.com/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/GenerateContent'
    AIStudioUrl: str = 'https://aistudio.google.com'
    WorkerCount: int = 1

    AioHTTPTimeout: int = 500

    UvicornHost: str = '0.0.0.0'
    UvicornPort: int = 8001
    AuthKey: str
    Proxy: ProxyConfig | None = None

    Headless: bool | Literal['virtual'] = 'virtual'
    Credentials: list[Credential] = Field(default_factory=list)


config = Config.model_validate(yaml.safe_load(open('config.yaml')))

if config.Debug:
    logging.basicConfig(level=logging.DEBUG)


# load states w/o credential
states = [fname for fname in os.listdir(config.StatesDir) if fname.endswith('.json')]
for credential in config.Credentials:
    if credential.stateFile is None:
        credential.stateFile = f'{credential.email}.json'
    if credential.stateFile in states:
        states.remove(credential.stateFile)

for stateFile in states:
    config.Credentials.append(Credential(
        email=stateFile[:-5],
        password=None,
        apikey=None,
        stateFile=f'{stateFile}',
    ))


# 验证代理配置
if config.Proxy:
    if config.Proxy.server.startswith('socks5://'):
        if config.Proxy.username:
            raise ValueError('Camoufox 不支持 SOCKS5 代理鉴权')
    elif config.Proxy.server.startswith('http://'):
        pass
    else:
        raise ValueError('不支持的代理配置')

AIOHTTP_PROXY = None
AIOHTTP_PROXY_AUTH = None
CAMOUFOX_PROXY = None

if config.Proxy:
    AIOHTTP_PROXY = config.Proxy.server
    CAMOUFOX_PROXY = {'server': config.Proxy.server}

    if config.Proxy.username is not None:
        AIOHTTP_PROXY_AUTH = aiohttp.BasicAuth(config.Proxy.username, config.Proxy.password or '')
        CAMOUFOX_PROXY['username'] = config.Proxy.username
        if config.Proxy.password is not None:
            CAMOUFOX_PROXY['password'] = config.Proxy.password
