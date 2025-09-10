import pydantic
from typing import Any, Optional, Union, List, Literal
from enum import Enum, IntEnum


class HarmCategory(str, Enum):
    HARM_CATEGORY_UNSPECIFIED = "HARM_CATEGORY_UNSPECIFIED"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_CIVIC_INTEGRITY = 'HARM_CATEGORY_CIVIC_INTEGRITY'


class HarmBlockThreshold(str, Enum):
    HARM_BLOCK_THRESHOLD_UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"
    OFF = "OFF"


class HarmBlockMethod(str, Enum):
    HARM_BLOCK_METHOD_UNSPECIFIED = "HARM_BLOCK_METHOD_UNSPECIFIED"
    SEVERITY = "SEVERITY"
    PROBABILITY = "PROBABILITY"


class Type(str, Enum):
    TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
    STRING = "STRING"
    NUMBER = "NUMBER"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    ARRAY = "ARRAY"
    OBJECT = "OBJECT"
    NULL = "NULL"


class Outcome(str, Enum):
    OUTCOME_UNSPECIFIED = "OUTCOME_UNSPECIFIED"
    OUTCOME_OK = "OUTCOME_OK"
    OUTCOME_FAILED = "OUTCOME_FAILED"
    OUTCOME_DEADLINE_EXCEEDED = "OUTCOME_DEADLINE_EXCEEDED"


class Language(str, Enum):
    LANGUAGE_UNSPECIFIED = "LANGUAGE_UNSPECIFIED"
    PYTHON = "PYTHON"


class FunctionResponseScheduling(str, Enum):
    SCHEDULING_UNSPECIFIED = "SCHEDULING_UNSPECIFIED"
    SILENT = "SILENT"
    WHEN_IDLE = "WHEN_IDLE"
    INTERRUPT = "INTERRUPT"


class FinishReason(str, Enum):
    FINISH_REASON_UNSPECIFIED = "FINISH_REASON_UNSPECIFIED"
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    LANGUAGE = "LANGUAGE"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"
    SPII = "SPII"
    MALFORMED_FUNCTION_CALL = "MALFORMED_FUNCTION_CALL"
    IMAGE_SAFETY = "IMAGE_SAFETY"
    UNEXPECTED_TOOL_CALL = "UNEXPECTED_TOOL_CALL"


class BlockedReason(str, Enum):
    BLOCKED_REASON_UNSPECIFIED = "BLOCKED_REASON_UNSPECIFIED"
    SAFETY = "SAFETY"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"


class HarmProbability(str, Enum):
    HARM_PROBABILITY_UNSPECIFIED = "HARM_PROBABILITY_UNSPECIFIED"
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HarmSeverity(str, Enum):
    HARM_SEVERITY_UNSPECIFIED = "HARM_SEVERITY_UNSPECIFIED"
    HARM_SEVERITY_NEGLIGIBLE = "HARM_SEVERITY_NEGLIGIBLE"
    HARM_SEVERITY_LOW = "HARM_SEVERITY_LOW"
    HARM_SEVERITY_MEDIUM = "HARM_SEVERITY_MEDIUM"
    HARM_SEVERITY_HIGH = "HARM_SEVERITY_HIGH"


class UrlRetrievalStatus(str, Enum):
    URL_RETRIEVAL_STATUS_UNSPECIFIED = "URL_RETRIEVAL_STATUS_UNSPECIFIED"
    URL_RETRIEVAL_STATUS_SUCCESS = "URL_RETRIEVAL_STATUS_SUCCESS"
    URL_RETRIEVAL_STATUS_ERROR = "URL_RETRIEVAL_STATUS_ERROR"


class FunctionCallingConfigMode(str, Enum):
    MODE_UNSPECIFIED = "MODE_UNSPECIFIED"
    AUTO = "AUTO"
    ANY = "ANY"
    NONE = "NONE"

class Behavior(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    BLOCKING = "BLOCKING"
    NON_BLOCKING = "NON_BLOCKING"

class MediaModality(str, Enum):
    MODALITY_UNSPECIFIED = "MODALITY_UNSPECIFIED"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    DOCUMENT = "DOCUMENT"

class TrafficType(str, Enum):
    TRAFFIC_TYPE_UNSPECIFIED = "TRAFFIC_TYPE_UNSPECIFIED"
    ON_DEMAND = "ON_DEMAND"
    PROVISIONED_THROUGHPUT = "PROVISIONED_THROUGHPUT"

class FileSource(str, Enum):
    SOURCE_UNSPECIFIED = "SOURCE_UNSPECIFIED"
    UPLOADED = "UPLOADED"
    GENERATED = "GENERATED"


# New Models
class GoogleTypeDate(pydantic.BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


class Citation(pydantic.BaseModel):
    startIndex: Optional[int] = None
    endIndex: Optional[int] = None
    uri: Optional[str] = None
    title: Optional[str] = None
    license: Optional[str] = None
    publicationDate: Optional[GoogleTypeDate] = None


class CitationMetadata(pydantic.BaseModel):
    citations: Optional[List[Citation]] = None


class SafetyRating(pydantic.BaseModel):
    category: HarmCategory
    probability: HarmProbability
    probabilityScore: Optional[float] = None
    severity: Optional[HarmSeverity] = None
    severityScore: Optional[float] = None
    blocked: Optional[bool] = None


class Segment(pydantic.BaseModel):
    partIndex: Optional[int] = None
    startIndex: Optional[int] = None
    endIndex: Optional[int] = None
    text: Optional[str] = None


class GroundingSupport(pydantic.BaseModel):
    segment: Optional[Segment] = None
    groundingChunkIndices: Optional[List[int]] = None
    confidenceScores: Optional[List[float]] = None


class RagChunkPageSpan(pydantic.BaseModel):
    firstPage: Optional[int] = None
    lastPage: Optional[int] = None


class RagChunk(pydantic.BaseModel):
    pageSpan: Optional[RagChunkPageSpan] = None
    text: Optional[str] = None


class GroundingChunkRetrievedContext(pydantic.BaseModel):
    uri: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    ragChunk: Optional[RagChunk] = None


class GroundingChunkWeb(pydantic.BaseModel):
    uri: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None


class GroundingChunk(pydantic.BaseModel):
    web: Optional[GroundingChunkWeb] = None
    retrievedContext: Optional[GroundingChunkRetrievedContext] = None


class RetrievalMetadata(pydantic.BaseModel):
    googleSearchDynamicRetrievalScore: Optional[float] = None


class SearchEntryPoint(pydantic.BaseModel):
    renderedContent: Optional[str] = None
    sdkBlob: Optional[bytes] = None


class GroundingMetadata(pydantic.BaseModel):
    retrievalQueries: Optional[List[str]] = None
    webSearchQueries: Optional[List[str]] = None
    groundingChunks: Optional[List[GroundingChunk]] = None
    groundingSupports: Optional[List[GroundingSupport]] = None
    retrievalMetadata: Optional[RetrievalMetadata] = None
    searchEntryPoint: Optional[SearchEntryPoint] = None


class LogprobsResultCandidate(pydantic.BaseModel):
    token: Optional[str] = None
    tokenId: Optional[int] = None
    logProbability: Optional[float] = None


class LogprobsResultTopCandidates(pydantic.BaseModel):
    candidates: Optional[List[LogprobsResultCandidate]] = None


class LogprobsResult(pydantic.BaseModel):
    chosenCandidates: Optional[List[LogprobsResultCandidate]] = None
    topCandidates: Optional[List[LogprobsResultTopCandidates]] = None


class UrlMetadata(pydantic.BaseModel):
    retrievedUrl: Optional[str] = None
    urlRetrievalStatus: Optional[UrlRetrievalStatus] = None


class UrlContextMetadata(pydantic.BaseModel):
    urlMetadata: Optional[List[UrlMetadata]] = None


class FunctionCallingConfig(pydantic.BaseModel):
    mode: Optional[FunctionCallingConfigMode] = None
    allowedFunctionNames: Optional[List[str]] = None


class LatLng(pydantic.BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class RetrievalConfig(pydantic.BaseModel):
    languageCode: Optional[str] = None
    latLng: Optional[LatLng] = None


class ToolConfig(pydantic.BaseModel):
    functionCallingConfig: Optional[FunctionCallingConfig] = None
    retrievalConfig: Optional[RetrievalConfig] = None

class ModalityTokenCount(pydantic.BaseModel):
    modality: Optional[MediaModality] = None
    tokenCount: Optional[int] = None

class FileStatus(pydantic.BaseModel):
    code: Optional[int] = None
    message: Optional[str] = None
    details: Optional[List[Any]] = None


# Tool and Schema related classes
class Schema(pydantic.BaseModel):
    type: Optional[Type] = None
    format: Optional[str] = None
    description: Optional[str] = None
    nullable: Optional[bool] = None
    items: Optional["Schema"] = None
    enum: Optional[List[str]] = None
    properties: Optional[dict[str, "Schema"]] = None
    required: Optional[List[str]] = None
    additionalProperties: Optional[Any] = None
    defs: Optional[dict[str, "Schema"]] = None
    ref: Optional[str] = None
    anyOf: Optional[List["Schema"]] = None
    default: Optional[Any] = None
    example: Optional[Any] = None
    maxItems: Optional[int] = None
    maxLength: Optional[int] = None
    maxProperties: Optional[int] = None
    maximum: Optional[float] = None
    minItems: Optional[int] = None
    minLength: Optional[int] = None
    minProperties: Optional[int] = None
    minimum: Optional[float] = None
    pattern: Optional[str] = None
    propertyOrdering: Optional[List[str]] = None
    title: Optional[str] = None


class FunctionDeclaration(pydantic.BaseModel):
    name: str
    description: str
    parameters: Optional[Schema] = None
    response: Optional[Schema] = None
    behavior: Optional[Behavior] = None


class Tool(pydantic.BaseModel):
    functionDeclarations: Optional[List[FunctionDeclaration]] = None
    retrieval: Optional[Any] = None
    googleSearch: Optional[Any] = None
    googleSearchRetrieval: Optional[Any] = None
    enterpriseWebSearch: Optional[Any] = None
    googleMaps: Optional[Any] = None
    urlContext: Optional[Any] = None
    codeExecution: Optional[Any] = None


# Core content classes
class VideoMetadata(pydantic.BaseModel):
    fps: Optional[float] = None
    endOffset: Optional[str] = None
    startOffset: Optional[str] = None


class ExecutableCode(pydantic.BaseModel):
    language: Language
    code: str


class CodeExecutionResult(pydantic.BaseModel):
    outcome: Outcome
    output: Optional[str] = None


class FunctionCall(pydantic.BaseModel):
    name: str
    args: Optional[dict[str, Any]] = None
    id: Optional[str] = None


class FunctionResponseContent(pydantic.BaseModel):
    content: dict[str, Any]


class FunctionResponse(pydantic.BaseModel):
    name: str
    response: dict[str, Any]
    willContinue: Optional[bool] = None
    scheduling: Optional[FunctionResponseScheduling] = None
    id: Optional[str] = None


class Blob(pydantic.BaseModel):
    data: str
    mimeType: str
    displayName: Optional[str] = None


class FileData(pydantic.BaseModel):
    fileUri: str
    mimeType: str
    displayName: Optional[str] = None


class Part(pydantic.BaseModel):
    text: Optional[str] = None
    inlineData: Optional[Blob] = None
    fileData: Optional[FileData] = None
    functionCall: Optional[FunctionCall] = None
    functionResponse: Optional[FunctionResponse] = None
    executableCode: Optional[ExecutableCode] = None
    codeExecutionResult: Optional[CodeExecutionResult] = None
    videoMetadata: Optional[VideoMetadata] = None
    thought: Optional[bool] = None
    thoughtSignature: Optional[bytes] = None


class Content(pydantic.BaseModel):
    parts: List[Part]
    role: Literal["user", "model", "system", "function"] | None = None


# Request/Response classes
class SafetySetting(pydantic.BaseModel):
    category: HarmCategory
    threshold: HarmBlockThreshold
    method: Optional[HarmBlockMethod] = None


class ThinkingConfig(pydantic.BaseModel):
    includeThoughts: Optional[bool] = None
    thinkingBudget: Optional[int] = None


class GenerateContentConfig(pydantic.BaseModel):
    temperature: Optional[float] = None
    topP: Optional[float] = None
    topK: Optional[float] = None
    candidateCount: Optional[int] = None
    maxOutputTokens: Optional[int] = None
    stopSequences: Optional[List[str]] = None
    responseMimeType: Optional[str] = None
    responseSchema: Optional[Schema] = None
    safetySettings: Optional[List[SafetySetting]] = None
    tools: Optional[List[Tool]] = None
    toolConfig: Optional[ToolConfig] = None
    # httpOptions: Optional[Any] = None
    systemInstruction: Optional[Content] = None
    responseLogprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    seed: Optional[int] = None
    routingConfig: Optional[Any] = None
    modelSelectionConfig: Optional[Any] = None
    labels: Optional[dict[str, str]] = None
    cachedContent: Optional[str] = None
    responseModalities: Optional[List[str]] = None
    mediaResolution: Optional[Any] = None
    speechConfig: Optional[Any] = None
    audioTimestamp: Optional[bool] = None
    automaticFunctionCalling: Optional[Any] = None
    thinkingConfig: Optional[ThinkingConfig] = None


class Candidate(pydantic.BaseModel):
    content: Content
    finishReason: Optional[FinishReason] = None
    index: int
    tokenCount: Optional[int] = None
    citationMetadata: Optional[CitationMetadata] = None
    finishMessage: Optional[str] = None
    safetyRatings: Optional[List[SafetyRating]] = None
    groundingMetadata: Optional[GroundingMetadata] = None
    logprobsResult: Optional[LogprobsResult] = None
    avgLogprobs: Optional[float] = None
    urlContextMetadata: Optional[UrlContextMetadata] = None


class GenerateContentResponsePromptFeedback(pydantic.BaseModel):
    blockReason: Optional[BlockedReason] = None
    blockReasonMessage: Optional[str] = None
    safetyRatings: Optional[List[SafetyRating]] = None


class UsageMetadata(pydantic.BaseModel):
    promptTokenCount: Optional[int] = 0
    candidatesTokenCount: Optional[int] = 0
    totalTokenCount: Optional[int] = 0
    cachedContentTokenCount: Optional[int] = None
    promptTokensDetails: Optional[List[ModalityTokenCount]] = None
    candidatesTokensDetails: Optional[List[ModalityTokenCount]] = None
    thoughtsTokenCount: Optional[int] = None
    toolUsePromptTokenCount: Optional[int] = None
    toolUsePromptTokensDetails: Optional[List[ModalityTokenCount]] = None
    trafficType: Optional[TrafficType] = None
    cacheTokensDetails: Optional[List[ModalityTokenCount]] = None


class GenerateContentResponse(pydantic.BaseModel):
    candidates: List[Candidate]
    promptFeedback: Optional[GenerateContentResponsePromptFeedback] = None
    usageMetadata: Optional[UsageMetadata] = None
    createTime: Optional[str] = None
    responseId: Optional[str] = None
    modelVersion: Optional[str] = None
    automaticFunctionCallingHistory: Optional[List[Content]] = None


class GenerateContentRequest(pydantic.BaseModel):
    contents: List[Content]
    safetySettings: Optional[List[SafetySetting]] = None
    generationConfig: Optional[GenerateContentConfig] = None
    tools: Optional[List[Tool]] = None
    systemInstruction: Optional[Content] = None


class File(pydantic.BaseModel):
    name: Optional[str] = None
    displayName: Optional[str] = None
    uri: Optional[str] = None
    state: Optional[str] = None
    mimeType: Optional[str] = None
    createTime: Optional[str] = None
    updateTime: Optional[str] = None
    expirationTime: Optional[str] = None
    sha256Hash: Optional[str] = None
    downloadUri: Optional[str] = None
    sizeBytes: Optional[int] = None
    source: Optional[FileSource] = None
    videoMetadata: Optional[dict[str, Any]] = None
    error: Optional[FileStatus] = None


class UploadFileRequest(pydantic.BaseModel):
    file: File


class FileResponse(pydantic.BaseModel):
    file: File


class Model(pydantic.BaseModel):
    name: str
    baseModelId: str
    version: str
    displayName: Optional[str] = None
    description: Optional[str] = None
    inputTokenLimit: Optional[int] = None
    outputTokenLimit: Optional[int] = None
    supportedGenerationMethods: Optional[List[str]] = None
    temperature: float
    topP: float
    topK: float


class ListModelsResponse(pydantic.BaseModel):
    models: Optional[List[Model]] = None
