from typing import Literal, Optional
import enum
import dataclasses
import json
from typing import AsyncIterator


class HarmCategory(enum.IntEnum):
    HARASSMENT = 7
    HATE = 8
    SEXUALLY_EXPLICIT = 9
    DANGEROUS_CONTENT = 10


class HarmBlockThreshold(enum.IntEnum):
    BLOCK_MOST = 1
    BLOCK_SOME = 2
    BLOCK_FEW = 3
    BLOCK_NONE = 4
    OFF = 5


@dataclasses.dataclass(kw_only=True)
class SafetySetting():
    unknow0: None = None
    unknow1: None = None
    category: HarmCategory
    threshold: HarmBlockThreshold


class SchemaType(enum.IntEnum):
    ENUM = 1
    STRING = 1
    NUMBER = 2
    INTEGER = 3
    BOOLEAN = 4
    ARRAY = 5
    OBJECT = 6

@dataclasses.dataclass(kw_only=True)
class PropertyItem():
    name: str
    schema: 'Schema'


@dataclasses.dataclass(kw_only=True)
class Schema():
    type: SchemaType
    unknow1: None = None
    unknow2: None = None
    unknow3: None = None
    enums: list[str] | None = None
    element: Optional['Schema'] = None
    properties: list[PropertyItem] | None = None
    required: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class FunctionDeclaration():
    name: str
    description: str
    parameters: Schema | None = None


class Modality(enum.IntEnum):
    UNSPECIFIED = 0
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    DOCUMENT = 4


@dataclasses.dataclass(kw_only=True)
class PromptHistoryConfig():
    unknow0: int = 1
    stopSequences: list[str] | None = None
    model: str
    unknow3: int | None = None
    topP: float = 0.95
    topK: int = 64
    maxOutputTokens: int = 65536
    safetySettings: list[SafetySetting]
    mimeType: Literal["text/plain", "application/json"] = "text/plain"
    codeExecution: int | None = None
    responseSchema: Schema | None = None
    functionDeclarations: list[FunctionDeclaration] | None = None
    unknow12: int | None = None
    unknow13: int | None = None
    googleSearch: int | None = None
    responseModalities: list[Modality] | None = None
    unknow16: int | None = None
    urlContext: int | None = None
    unknow18: int | None = None
    unknow19: int | None = None
    unknow20: int | None = None
    unknow21: int | None = None
    unknow22: int | None = None
    unknow23: int | None = None
    thinkingBudget: int | None = None
    googleSearchRetrieval: list[int] | None = None


@dataclasses.dataclass(kw_only=True)
class UserInfo():
    name: str
    unknow1: int = 1
    avatar: str


@dataclasses.dataclass(kw_only=True)
class PromptMetadata():
    title: str
    unknow1: None = None
    user: UserInfo
    unknow3: None = None
    unknow4: tuple[tuple[str, int], UserInfo]
    unknow5: tuple[int, int, int] = (1, 1, 1)
    unknow6: None = None
    unknow7: None = None
    unknow8: None = None
    unknow9: None = None
    unknow10: list[None] | None = None


@dataclasses.dataclass(kw_only=True)
class FunctionCallParameterArray():
    elements: list['FunctionCallParameter'] | None = None


@dataclasses.dataclass(kw_only=True)
class FunctionCallParameter():
    un0: None = None
    number: float | int | None = None
    string: str | None = None  # ENUM
    isBool: bool | None = None
    object: list[tuple[str, 'FunctionCallParameter']] | None = None
    array: FunctionCallParameterArray | None = None


@dataclasses.dataclass(kw_only=True)
class FunctionCallArg():
    key: str
    value: FunctionCallParameter


@dataclasses.dataclass(kw_only=True)
class FunctionCall():
    name: str
    args: tuple[list[FunctionCallArg]] | None = None


@dataclasses.dataclass(kw_only=True)
class FunctionCallRecord():
    functionCall: FunctionCall
    response: str | None = None

@dataclasses.dataclass(kw_only=True)
class Blob():
    mimeType: str
    data: str # base64encoded
    displayName: Optional[str] = None


@dataclasses.dataclass(kw_only=True)
class PromptContent():
    text: str | None = None
    imageId: None = None
    videoId: None = None
    fileId: tuple[str] | None = None
    unknow4: None = None
    audio: None = None
    image: None = None
    video: None = None
    role: Literal["user", "model"]
    unknow9: None = None
    codeExecutionData: None = None
    codeExecutionResult: None = None
    generatedImage: Blob | None = None
    youtube: None = None
    unknow14: None = None
    unknow15: int | None = None
    isModel: int | None = None  #?
    generatedAudio: None = None
    tokens: int | None = None
    unknow19: None = None
    functionCall: FunctionCallRecord | None = None
    unknow21: None = None
    unknow22: None = None
    generatedFile: Blob | None = None
    unknow24: None = None
    unknow25: None = None
    unknow26: None = None
    thoughtSignature: list[str] | None = None #?


@dataclasses.dataclass(kw_only=True)
class PromptRecord():
    history: list[PromptContent]
    input: list[PromptContent] | None = None


@dataclasses.dataclass(kw_only=True)
class PromptInfo():
    uri: str
    unknow1: int | None = None
    unknow2: int | None = None
    generationConfig: PromptHistoryConfig
    promptMetadata: PromptMetadata
    unknow5: None = None
    unknow6: None = None
    unknow7: None = None
    unknow8: None = None
    unknow9: None = None
    unknow10: None = None
    unknow11: None = None
    systemInstruction: list[str] | None = None
    contents: PromptRecord


@dataclasses.dataclass(kw_only=True)
class PromptHistory():
    prompt: PromptInfo


@dataclasses.dataclass(kw_only=True)
class GenerateUsage():
    inputToken: int | None = 0
    outputTokens: int | None = 0
    totalTokens: int | None = 0
    unknow3: None = None
    unknow4: list[tuple[int, int]] | None = None
    unknow5: None = None
    unknow6: None = None
    unknow7: None = None
    unknow8: None = None
    reasoningTokens: int | None = 0


class Language(enum.IntEnum):
    Python = 1


class Outcome(enum.IntEnum):
    OK = 1
    FAILED = 2
    DEADLINE_EXCEEDED = 3


@dataclasses.dataclass(kw_only=True)
class ExecutableCode():
    language: Language | None
    code: str


@dataclasses.dataclass(kw_only=True)
class CodeExecutionResult():
    outcome: Outcome | None = None
    output: str


@dataclasses.dataclass(kw_only=True)
class GeneratePart():
    unknow0: None = None
    text: str | None = None
    inlineData: Blob | None = None
    unknow3: None = None
    unknow4: None = None
    unknow5: None = None
    unknow6: None = None
    executable_code: ExecutableCode | None = None
    code_execution_result: CodeExecutionResult | None = None
    unknow9: None = None
    functionCall: FunctionCall | None = None
    unknow11: None = None
    isThought: int | None = None
    unknow13: None = None
    thoughtSignature: str | None = None


@dataclasses.dataclass(kw_only=True)
class GenerateContent():
    parts: list[GeneratePart] | None = None
    role: Literal['model', 'user']


@dataclasses.dataclass(kw_only=True)
class Segment():
    partIndex: int | None = None
    startIndex: int | None = None
    endIndex: int | None = None
    text: str | None = None


@dataclasses.dataclass(kw_only=True)
class GroundingSupport():
    segment: Segment | None = None
    groundingChunkIndices: list[int] | None = None
    confidenceScores: list[float] | None = None


@dataclasses.dataclass(kw_only=True)
class GroundingChunkWeb():
    url: str | None = None
    title: str | None = None
    domain: str | None = None


@dataclasses.dataclass(kw_only=True)
class GroundingChunk():
    web: GroundingChunkWeb | None = None


@dataclasses.dataclass(kw_only=True)
class GroundingMetadata():
    searchEntryPoint: tuple[str] | None = None
    groundingChunks: list[GroundingChunk] | None = None
    groundingSupports: list[GroundingSupport] | None = None
    unknow3: int | None = None
    webSearchQueries: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class Candidate():
    contents: GenerateContent | None = None
    isOutput: int | None = None # 1 response 2 thinking
    finishReason: str | None = None
    unknow3: None = None
    unknow4: None = None
    unknow5: None = None
    unknow6: None = None
    groundingMetadata: GroundingMetadata | None = None


@dataclasses.dataclass(kw_only=True)
class StreamEvent():
    candidates: list[Candidate] | None = None
    unknow1: None = None
    usage: GenerateUsage | None = None
    unknow4: tuple[str, int, int] | None = None
    unknow5: None = None
    unknow6: None = None
    unknow7: None = None


@dataclasses.dataclass(kw_only=True)
class ResponseErrorDetail():
    exception: str
    sources: list[tuple[str, str] | None] | None = None


@dataclasses.dataclass(kw_only=True)
class ResponseError(BaseException):
    code: int
    message: str
    details: list[ResponseErrorDetail] | None = None

    def __str__(self) -> str:
        return f'<AIStudio Response Error [{self.code}]{self.message}'

@dataclasses.dataclass(kw_only=True)
class GenerateContentResponse():
    events: list[StreamEvent]
    error: ResponseError | None = None


@dataclasses.dataclass(kw_only=True)
class ThinkingConfig():
    includeThoughts: int = 1
    thinkingBudget: int = -1


@dataclasses.dataclass(kw_only=True)
class GenerateContentConfig():
    unknow0: None = None
    unknow1: None = None
    unknow2: None = None
    maxOutputTokens: int = 65536
    temperature: float = 0.95
    topP: float = 1.0
    topK: int = 64
    responseMimeType: str | None = None
    responseSchema: None = None
    unknow9: None = None
    unknow10: None = None
    unknow11: None = None
    unknow12: None = None
    unknow13: int = 1
    unknow14: None = None
    unknow15: None = None
    thinkingConfig: ThinkingConfig
    mediaResolution: int | None = None # ?


@dataclasses.dataclass(kw_only=True)
class FunctionDeclarationWrap():
    functionDeclarations: list[FunctionDeclaration]

@dataclasses.dataclass(kw_only=True)
class Tool():
    functionDeclarations: FunctionDeclarationWrap


@dataclasses.dataclass(kw_only=True)
class GenerateContentRequest():
    model: str
    contents: list[GenerateContent]
    safetySettings: list[SafetySetting] | None = None
    generationConfig: GenerateContentConfig | None = None
    potoken: str
    systemInstruction: GenerateContent | None = None
    tools: list[Tool] | None = None
    unknow7: None = None
    unknow8: None = None
    unknow9: None = None
    unknow10: int = 1
    unknow11: str | None = None


@dataclasses.dataclass(kw_only=True)
class Model():
    name: str
    unknow2: None = None
    version: str
    displayName: str
    description: str | None = None
    inputTokenLimit: int
    outputTokenLimit: int
    supportedActions: list[str] | None = None
    temperature: float | None = None
    topK: float | None = None
    topP: float | None = None


@dataclasses.dataclass(kw_only=True)
class ListModelsResponse():
    models: list[Model]

import typing
import types
import dataclasses


FlattenData = int | float | str | None | list['FlattenData']


def convertdataclass[T](data: list[FlattenData], dataType: type[T]) -> T:
    assert dataclasses.is_dataclass(dataType)
    params = {}
    if data is None:
        return None # type: ignore
    for (key, fieldType), value in zip(typing.get_type_hints(dataType).items(), data):
        isOptional = False
        if fieldType is types.NoneType or fieldType is None:
            isOptional = True
        elif typing.get_origin(fieldType) in (types.UnionType, typing.Union):
            args = list(typing.get_args(fieldType))
            if types.NoneType in args:
                isOptional = True
                args.remove(types.NoneType)
            if len(args) == 1:
                fieldType = args[0]
            else:
                fieldType = tuple(args)
        if value is None:
            assert isOptional, ValueError(f'field {key} is required')
            continue
        params[key] = inflate(value, fieldType) # type: ignore
    return dataType(**params) # type: ignore


def inflate[T](value: typing.Any, dataType: type[T]) -> T | None:
    if isinstance(dataType, tuple):
        return value
    if dataclasses.is_dataclass(dataType):
        rtn = convertdataclass(value, dataType)
        if rtn is None:
            return None
        return rtn
    orig = typing.get_origin(dataType)
    if orig is list:
        args = typing.get_args(dataType)
        assert len(args) == 1
        if not isinstance(value, list):
            return value
        elementType = args[0]
        return [
            inflate(item, elementType)
            for item in value
        ]
    elif orig is tuple:
        args = typing.get_args(dataType)
        return [
            inflate(item, itemType)
            for item, itemType in zip(value, args)
        ]
    elif orig is None and isinstance(dataType, enum.EnumType):
        return dataType(value)
    else:
        return value


def flatten(data: typing.Any) -> FlattenData:
    rtn: list[FlattenData]
    if dataclasses.is_dataclass(data):
        rtn = [flatten(v) for k, v in dataclasses.asdict(data).items()]
    elif isinstance(data, dict):
        rtn = [flatten(v) for k, v in data.items()]
    elif isinstance(data, (list, tuple)):
        rtn = [flatten(v) for v in data]
    elif isinstance(data, (enum.Enum, )):
        return data.value
    elif isinstance(data, (int, float, str)) or data is None:
        return data
    else:
        return data
    while rtn and rtn[-1] is None: rtn.pop()
    return rtn


import json
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, enum.Enum):
            return o.name
        return super().default(o)


from typing import AsyncIterator, AsyncGenerator


async def StreamParser(source: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
    # 流式解析GenerateContentResponse并在StreamEvent可用时立刻yield出
    # 实现方式是只考虑[]\",的简易词法分析器
    stack = []
    buff = bytearray()
    idx = 0

    async for chunk in source:
        buff.extend(chunk)
        pos = 0
        escape = False
        consumed = (len(buff), 0)
        for pos, c in enumerate(buff):
            if stack and pos <= stack[-1][0]:
                continue
            if escape:
                escape = False
                continue
            if escape := (c == 92):  # \
                continue
            if c == 34:  # "
                if stack and stack[-1][-1] == 34:
                    stack.pop()
                else:
                    stack.append((pos, c))
            if stack and stack[-1][-1] == 34:
                continue
            if c == 91:  # [
                stack.append((pos, c))
            if c == 93:  # ]
                begin, _ = stack.pop()
                if len(stack) == 2 and (stack[-1][0] - stack[-2][0]) == 1:
                    event = inflate(json.loads(buff[begin: pos + 1]), StreamEvent)
                    yield event
                    if buff[begin - 1] == 44: # ,
                        begin -= 1
                    consumed = (min(begin, consumed[0]), max(pos + 1, consumed[1]))
                elif len(stack) == 1 and buff[begin - 1] == 44:
                    data = json.loads(buff[begin: pos + 1])
                    if data and (error := inflate(data, ResponseError)):
                        raise error
        b, e = consumed
        if e >= b:
            del buff[b: e]
