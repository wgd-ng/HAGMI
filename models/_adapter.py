import random
import base64
import dataclasses
import json
from typing import NoReturn, List, Any
from .genai import GenerateContentRequest, GenerateContentResponse, UsageMetadata
from .aistudio import PromptHistory, StreamEvent
from . import aistudio, genai


HARM_CATEGORY_MAP = {
    genai.HarmCategory.HARM_CATEGORY_HATE_SPEECH: aistudio.HarmCategory.HATE,
    genai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: aistudio.HarmCategory.SEXUALLY_EXPLICIT,
    genai.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: aistudio.HarmCategory.DANGEROUS_CONTENT,
    genai.HarmCategory.HARM_CATEGORY_HARASSMENT: aistudio.HarmCategory.HARASSMENT,
}

HARM_BLOCK_THRESHOLD_MAP = {
    genai.HarmBlockThreshold.BLOCK_NONE: aistudio.HarmBlockThreshold.BLOCK_NONE,
    genai.HarmBlockThreshold.BLOCK_ONLY_HIGH: aistudio.HarmBlockThreshold.BLOCK_FEW,
    genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: aistudio.HarmBlockThreshold.BLOCK_SOME,
    genai.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE: aistudio.HarmBlockThreshold.BLOCK_MOST,
    genai.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED: aistudio.HarmBlockThreshold.OFF, # Default mapping
}

GENAI_TYPE_TO_AISTUDIO_SCHEMATYPE = {
    genai.Type.STRING: aistudio.SchemaType.STRING,
    genai.Type.NUMBER: aistudio.SchemaType.NUMBER,
    genai.Type.INTEGER: aistudio.SchemaType.INTEGER,
    genai.Type.BOOLEAN: aistudio.SchemaType.BOOLEAN,
    genai.Type.ARRAY: aistudio.SchemaType.ARRAY,
    genai.Type.OBJECT: aistudio.SchemaType.OBJECT,
}

GENAI_MODALITY_TO_AISTUDIO_MAP = {
    "TEXT": aistudio.Modality.TEXT,
    "IMAGE": aistudio.Modality.IMAGE,
    "AUDIO": aistudio.Modality.AUDIO,
    "DOCUMENT": aistudio.Modality.DOCUMENT
}


def _randomPromptId() -> str:
    return base64.b64encode(random.randbytes(24), altchars=b'-_').decode()


def GenAISchemaToAiStudioSchema(schema: genai.Schema) -> aistudio.Schema:
    """Converts a GenAI Schema to an AiStudio Schema."""
    schema_type = schema.type
    if not schema_type:
        # Default to object if type is not specified. This can happen for empty parameter objects.
        schema_type = genai.Type.OBJECT

    aistudio_schema = aistudio.Schema(
        type=GENAI_TYPE_TO_AISTUDIO_SCHEMATYPE[schema_type]
    )

    if schema.properties:
        aistudio_schema.properties = [
            aistudio.PropertyItem(name=name, schema=GenAISchemaToAiStudioSchema(prop_schema))
            for name, prop_schema in schema.properties.items()
        ]

    if schema.items:
        aistudio_schema.element = GenAISchemaToAiStudioSchema(schema.items)

    if schema.enum:
        aistudio_schema.enums = schema.enum

    if schema.required:
        aistudio_schema.required = schema.required

    return aistudio_schema


def AIStudioFunctionCallParameterToGenAI(param: aistudio.FunctionCallParameter) -> Any:
    if param.number is not None:
        return param.number
    if param.string is not None:
        return param.string
    if param.isBool is not None:
        return param.isBool
    if param.array is not None and param.array.elements is not None:
        return [AIStudioFunctionCallParameterToGenAI(p) for p in param.array.elements]
    if param.object is not None:
        return {k: AIStudioFunctionCallParameterToGenAI(v) for k, v in param.object}
    return None


def AIStudioFunctionCallToGenAI(call: aistudio.FunctionCall) -> genai.FunctionCall:
    args: dict[str, Any] | None = None
    if call.args:
        args = {}
        for arg in call.args[0]:
            args[arg.key] = AIStudioFunctionCallParameterToGenAI(arg.value)
    return genai.FunctionCall(
        name=call.name,
        args=args,
    )


def _GenAIAnyToAIStudioFunctionCallParameter(value: Any) -> aistudio.FunctionCallParameter:
    if isinstance(value, str):
        return aistudio.FunctionCallParameter(string=value)
    if isinstance(value, bool):
        return aistudio.FunctionCallParameter(isBool=value)
    if isinstance(value, (int, float)):
        return aistudio.FunctionCallParameter(number=value)
    if isinstance(value, list):
        return aistudio.FunctionCallParameter(array=[_GenAIAnyToAIStudioFunctionCallParameter(v) for v in value])
    if isinstance(value, dict):
        return aistudio.FunctionCallParameter(object=[(k, _GenAIAnyToAIStudioFunctionCallParameter(v)) for k, v in value.items()])
    raise NotImplementedError(f"Unsupported type for FunctionCall arg: {type(value)}")


def GenAIFunctionCallToAIStudio(call: genai.FunctionCall) -> aistudio.FunctionCall:
    final_args: tuple[list[aistudio.FunctionCallArg]] | None = None
    if call.args:
        args: list[aistudio.FunctionCallArg] = []
        for key, value in call.args.items():
            args.append(aistudio.FunctionCallArg(
                key=key,
                value=_GenAIAnyToAIStudioFunctionCallParameter(value)
            ))
        if args:
            final_args = (args,)

    return aistudio.FunctionCall(
        name=call.name,
        args=final_args,
    )


def GenAIFunctionDeclarationToAiStudio(func: genai.FunctionDeclaration) -> aistudio.FunctionDeclaration:
    parameters = None
    if func.parameters:
        parameters = GenAISchemaToAiStudioSchema(func.parameters)

    return aistudio.FunctionDeclaration(
        name=func.name,
        description=func.description,
        parameters=parameters,
    )

def GenAIRequestToAiStudioPromptHistory(model: str, request: GenerateContentRequest, prompt_id: str | None = None) -> PromptHistory:
    if prompt_id is None:
        prompt_id = _randomPromptId()

    system_instruction = []
    if request.systemInstruction:
        system_instruction = [part.text for part in request.systemInstruction.parts if part.text]

    contents = request.contents

    turns: list[aistudio.PromptContent] = []

    for content in contents:
        for part in content.parts:
            if content.role == 'system' and part.text:
                system_instruction.append(part.text)
            elif content.role in ('model', 'user'):
                if part.functionCall and part.functionCall:
                    turns.append(aistudio.PromptContent(
                        role='model',
                        functionCall=aistudio.FunctionCallRecord(
                            functionCall=GenAIFunctionCallToAIStudio(part.functionCall),
                            response=None
                        ),
                    ))
                if part.text:
                    turns.append(aistudio.PromptContent(role=content.role, text=part.text))
                if part.inlineData:
                    # TODO: dispatch inline data base on mimeType
                    turns.append(aistudio.PromptContent(
                        role='user',
                        generatedFile=aistudio.Blob(
                            mimeType=part.inlineData.mimeType,
                            data=part.inlineData.data,
                            displayName='inlineData'
                        )
                    ))
                if part.functionResponse:
                    for turn in reversed(turns):
                        if not turn.functionCall:
                            continue
                        if turn.functionCall.functionCall.name != part.functionResponse.name:
                            continue
                        response = part.functionResponse.response
                        if isinstance(response, str):
                            turn.functionCall.response = response
                        elif isinstance(response, dict) and 'output' in response:
                            turn.functionCall.response = response['output'] if isinstance(response['output'], str) else json.dumps(response['output'])
                        else:
                            turn.functionCall.response = json.dumps(response)

    for turn in turns[:-1]:
        if turn.functionCall and not turn.functionCall.response:
            turn.functionCall.response = '(missing response)'

    turns.append(aistudio.PromptContent(role='model', text='(placeholder)'))

    safety_settings = []
    if request.safetySettings:
        for s in request.safetySettings:
            if s.category in HARM_CATEGORY_MAP:
                safety_settings.append(aistudio.SafetySetting(
                    category=HARM_CATEGORY_MAP[s.category],
                    threshold=HARM_BLOCK_THRESHOLD_MAP.get(s.threshold, aistudio.HarmBlockThreshold.OFF)
                ))

    if request.generationConfig:
        generation_config = aistudio.GenerateContentConfig(
            model=f'models/{model}',
            safetySettings=safety_settings,
            topP=request.generationConfig.topP if request.generationConfig.topP is not None else 0.95,
            topK=int(request.generationConfig.topK) if request.generationConfig.topK is not None else 64,
            maxOutputTokens=request.generationConfig.maxOutputTokens if request.generationConfig.maxOutputTokens is not None else 65536,
            stopSequences=request.generationConfig.stopSequences if request.generationConfig.stopSequences is not None else None
        )
        if request.generationConfig.responseSchema:
            generation_config.responseSchema = GenAISchemaToAiStudioSchema(request.generationConfig.responseSchema)
            generation_config.mimeType = "application/json"
        if request.generationConfig.thinkingConfig:
            generation_config.thinkingBudget = request.generationConfig.thinkingConfig.thinkingBudget
        if request.generationConfig.responseModalities:
            generation_config.responseModalities = [
                GENAI_MODALITY_TO_AISTUDIO_MAP[item] for item in request.generationConfig.responseModalities
            ]
    else:
        generation_config = aistudio.GenerateContentConfig(
            model=f'models/{model}',
            safetySettings=safety_settings
        )

    if request.tools:
        generation_config.functionDeclarations = []
        for tool in request.tools:
            if tool.functionDeclarations:
                generation_config.functionDeclarations.extend([
                    GenAIFunctionDeclarationToAiStudio(func)
                    for func in tool.functionDeclarations
                ])
            if tool.codeExecution:
                generation_config.codeExecution = 1
            if tool.googleSearch:
                generation_config.googleSearch = 1
            if tool.urlContext:
                generation_config.urlContext = 1

    # Creating dummy user info as it's required but not available in the request
    user_info = aistudio.UserInfo(name="user", avatar="")

    prompt_metadata = aistudio.PromptMetadata(
        title=f"{prompt_id}",
        user=user_info,
        unknow4=(('', 0), user_info)
    )

    prompt = aistudio.PromptInfo(
        uri=f"prompts/{prompt_id}",
        generationConfig=generation_config,
        promptMetadata=prompt_metadata,
        systemInstruction=system_instruction,
        contents=aistudio.PromptRecord(
            history=turns,
        ),
    )

    return aistudio.PromptHistory(prompt=prompt)


def AiStudioStreamEventToGenAIResponse(events: StreamEvent | List[StreamEvent]) -> GenerateContentResponse:
    if not isinstance(events, list):
        events = [events]

    candidates = []
    usage = None
    for event in events:
        if event.usage:
            usage = event.usage

        if event.candidates:
            for candidate in event.candidates:
                finish_reason = None
                parts = []
                role = 'model'
                # if not candidate.contents.parts:
                #     finish_reason = genai.FinishReason.STOP
                # else:
                if candidate.contents and candidate.contents.parts:
                    for part in candidate.contents.parts:
                        # TODO: 添加其他返回类型支持
                        if part.text:
                            parts.append(genai.Part(
                                text=part.text,
                                thought=bool(part.isThought),
                            ))
                        if part.inlineData:
                            parts.append(genai.Part(
                                inlineData=genai.Blob(
                                    mimeType=part.inlineData.mimeType,
                                    data=part.inlineData.data,
                                    displayName=part.inlineData.displayName,
                                )
                            ))
                        if part.functionCall:
                            parts.append(genai.Part(
                                functionCall=AIStudioFunctionCallToGenAI(part.functionCall)
                            ))
                        # TODO: Structed output
                        # TODO: Grounding
                        # TODO: URL Context
                        # TODO: Code Execution
                candidates.append(genai.Candidate(
                    content=genai.Content(
                        parts=parts,
                        role=role,
                    ),
                    finishReason=finish_reason,
                    index=0,
                    tokenCount=0,
                ))

    return genai.GenerateContentResponse(
        candidates=candidates,
        usageMetadata=genai.UsageMetadata(
            promptTokenCount=usage.inputToken or 0,
            candidatesTokenCount=usage.outputTokens or 0,
            totalTokenCount=usage.totalTokens or 0,
            thoughtsTokenCount=usage.reasoningTokens or 0,
            toolUsePromptTokenCount=0,
            cachedContentTokenCount=0,
        ) if usage else None,
    )


def AIStudioListModelToGenAIListModel(models: aistudio.ListModelsResponse) -> genai.ListModelsResponse:
    genai_models = []
    for model in models.models:
        genai_models.append(genai.Model(
            name=model.name,
            baseModelId=model.name.split('/')[-1] if '/' in model.name else model.name,
            version=model.version,
            displayName=model.displayName,
            description=model.description,
            inputTokenLimit=model.inputTokenLimit,
            outputTokenLimit=model.outputTokenLimit,
            supportedGenerationMethods=model.supportedActions,
            temperature=model.temperature if model.temperature is not None else 1.0,
            topP=model.topP if model.topP is not None else 0.95,
            topK=float(model.topK) if model.topK is not None else 64.0,
        ))
    return genai.ListModelsResponse(models=genai_models)
