import os
import logging
from pydantic import BaseModel, config
import yaml


class Credential(BaseModel):
    email: str | None = None
    password: str | None = None
    apikey: str | None = None
    stateFile: str | None = None


class Config(BaseModel):
    Debug: bool = False
    LogDir: str = './gemini_logs'
    StatesDir: str = './states'
    AIStudioAPIUrl: str = 'https://alkalimakersuite-pa.clients6.google.com/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService/GenerateContent'
    AIStudioUrl: str = 'https://aistudio.google.com'
    AIStudioProxy: str | None = None
    WorkerCount: int = 1

    AioHTTPTimeout: int = 500

    UvicornHost: str = '0.0.0.0'
    UvicornPort: int = 8001
    AuthKey: str

    Headless: bool | str = 'virtual'
    Credentials: list[Credential]


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
