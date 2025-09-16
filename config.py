import os
import logging
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


class ModelInjection(BaseModel):
    model: str
    template: str
    override: dict[str, str | int] = Field(default_factory=dict)


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

    Headless: bool | str = 'virtual'
    Credentials: list[Credential] = Field(default_factory=list)
    CustomModels: list[ModelInjection] = Field(default_factory=list)


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
