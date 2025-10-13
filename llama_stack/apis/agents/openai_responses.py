# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from llama_stack.apis.vector_io import SearchRankingOptions as FileSearchRankingOptions
from llama_stack.schema_utils import json_schema_type, register_schema

# NOTE(ashwin): this file is literally a copy of the OpenAI responses API schema. We should probably
# take their YAML and generate this file automatically. Their YAML is available.


@json_schema_type
class OpenAIResponseError(BaseModel):
    """Error details for failed OpenAI response requests.

    :param code: Error code identifying the type of failure
    :param message: Human-readable error message describing the failure
    """

    code: str
    message: str


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    """Text content for input messages in OpenAI response format.

    :param text: The text content of the input message
    :param type: Content type identifier, always "input_text"
    """

    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    """Image content for input messages in OpenAI response format.

    :param detail: Level of detail for image processing, can be "low", "high", or "auto"
    :param type: Content type identifier, always "input_image"
    :param image_url: (Optional) URL of the image content
    """

    detail: Literal["low"] | Literal["high"] | Literal["auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    # TODO: handle file_id
    image_url: str | None = None


# TODO: handle file content types
OpenAIResponseInputMessageContent = Annotated[
    OpenAIResponseInputMessageContentText | OpenAIResponseInputMessageContentImage,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")


@json_schema_type
class OpenAIResponseAnnotationFileCitation(BaseModel):
    """File citation annotation for referencing specific files in response content.

    :param type: Annotation type identifier, always "file_citation"
    :param file_id: Unique identifier of the referenced file
    :param filename: Name of the referenced file
    :param index: Position index of the citation within the content
    """

    type: Literal["file_citation"] = "file_citation"
    file_id: str
    filename: str
    index: int


@json_schema_type
class OpenAIResponseAnnotationCitation(BaseModel):
    """URL citation annotation for referencing external web resources.

    :param type: Annotation type identifier, always "url_citation"
    :param end_index: End position of the citation span in the content
    :param start_index: Start position of the citation span in the content
    :param title: Title of the referenced web resource
    :param url: URL of the referenced web resource
    """

    type: Literal["url_citation"] = "url_citation"
    end_index: int
    start_index: int
    title: str
    url: str


@json_schema_type
class OpenAIResponseAnnotationContainerFileCitation(BaseModel):
    type: Literal["container_file_citation"] = "container_file_citation"
    container_id: str
    end_index: int
    file_id: str
    filename: str
    start_index: int


@json_schema_type
class OpenAIResponseAnnotationFilePath(BaseModel):
    type: Literal["file_path"] = "file_path"
    file_id: str
    index: int


OpenAIResponseAnnotations = Annotated[
    OpenAIResponseAnnotationFileCitation
    | OpenAIResponseAnnotationCitation
    | OpenAIResponseAnnotationContainerFileCitation
    | OpenAIResponseAnnotationFilePath,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseAnnotations, name="OpenAIResponseAnnotations")


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)


@json_schema_type
class OpenAIResponseContentPartRefusal(BaseModel):
    """Refusal content within a streamed response part.

    :param type: Content part type identifier, always "refusal"
    :param refusal: Refusal text supplied by the model
    """

    type: Literal["refusal"] = "refusal"
    refusal: str


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText | OpenAIResponseContentPartRefusal,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseMessage(BaseModel):
    """
    Corresponds to the various Message types in the Responses API.
    They are all under one type because the Responses API gives them all
    the same "type" value, and there is no way to tell them apart in certain
    scenarios.
    """

    content: str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent]
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"]
    type: Literal["message"] = "message"

    # The fields below are not used in all scenarios, but are required in others.
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    """Web search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param status: Current status of the web search operation
    :param type: Tool call type identifier, always "web_search_call"
    """

    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


class OpenAIResponseOutputMessageFileSearchToolCallResults(BaseModel):
    """Search results returned by the file search operation.

    :param attributes: (Optional) Key-value attributes associated with the file
    :param file_id: Unique identifier of the file containing the result
    :param filename: Name of the file containing the result
    :param score: Relevance score for this search result (between 0 and 1)
    :param text: Text content of the search result
    """

    attributes: dict[str, Any]
    file_id: str
    filename: str
    score: float
    text: str


@json_schema_type
class OpenAIResponseOutputMessageFileSearchToolCall(BaseModel):
    """File search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param queries: List of search queries executed
    :param status: Current status of the file search operation
    :param type: Tool call type identifier, always "file_search_call"
    :param results: (Optional) Search results returned by the file search operation
    """

    id: str
    queries: list[str]
    status: str
    type: Literal["file_search_call"] = "file_search_call"
    results: list[OpenAIResponseOutputMessageFileSearchToolCallResults] | None = None


@json_schema_type
class OpenAIResponseOutputMessageFunctionToolCall(BaseModel):
    """Function tool call output message for OpenAI responses.

    :param call_id: Unique identifier for the function call
    :param name: Name of the function being called
    :param arguments: JSON string containing the function arguments
    :param type: Tool call type identifier, always "function_call"
    :param id: (Optional) Additional identifier for the tool call
    :param status: (Optional) Current status of the function call execution
    """

    call_id: str
    name: str
    arguments: str
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPCall(BaseModel):
    """Model Context Protocol (MCP) call output message for OpenAI responses.

    :param id: Unique identifier for this MCP call
    :param type: Tool call type identifier, always "mcp_call"
    :param arguments: JSON string containing the MCP call arguments
    :param name: Name of the MCP method being called
    :param server_label: Label identifying the MCP server handling the call
    :param error: (Optional) Error message if the MCP call failed
    :param output: (Optional) Output result from the successful MCP call
    """

    id: str
    type: Literal["mcp_call"] = "mcp_call"
    arguments: str
    name: str
    server_label: str
    error: str | None = None
    output: str | None = None


class MCPListToolsTool(BaseModel):
    """Tool definition returned by MCP list tools operation.

    :param input_schema: JSON schema defining the tool's input parameters
    :param name: Name of the tool
    :param description: (Optional) Description of what the tool does
    """

    input_schema: dict[str, Any]
    name: str
    description: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPListTools(BaseModel):
    """MCP list tools output message containing available tools from an MCP server.

    :param id: Unique identifier for this MCP list tools operation
    :param type: Tool call type identifier, always "mcp_list_tools"
    :param server_label: Label identifying the MCP server providing the tools
    :param tools: List of available tools provided by the MCP server
    """

    id: str
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: str
    tools: list[MCPListToolsTool]


@json_schema_type
class OpenAIResponseMCPApprovalRequest(BaseModel):
    """
    A request for human approval of a tool invocation.
    """

    arguments: str
    id: str
    name: str
    server_label: str
    type: Literal["mcp_approval_request"] = "mcp_approval_request"


@json_schema_type
class OpenAIResponseMCPApprovalResponse(BaseModel):
    """
    A response to an MCP approval request.
    """

    approval_request_id: str
    approve: bool
    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    id: str | None = None
    reason: str | None = None


OpenAIResponseOutput = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools
    | OpenAIResponseMCPApprovalRequest,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


# This has to be a TypedDict because we need a "schema" field and our strong
# typing code in the schema generator doesn't support Pydantic aliases. That also
# means we can't use a discriminator field here, because TypedDicts don't support
# default values which the strong typing code requires for discriminators.
class OpenAIResponseTextFormat(TypedDict, total=False):
    """Configuration for Responses API text format.

    :param type: Must be "text", "json_schema", or "json_object" to identify the format type
    :param name: The name of the response format. Only used for json_schema.
    :param schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model. Only used for json_schema.
    :param description: (Optional) A description of the response format. Only used for json_schema.
    :param strict: (Optional) Whether to strictly enforce the JSON schema. If true, the response must match the schema exactly. Only used for json_schema.
    """

    type: Literal["text"] | Literal["json_schema"] | Literal["json_object"]
    name: str | None
    schema: dict[str, Any] | None
    description: str | None
    strict: bool | None


@json_schema_type
class OpenAIResponseText(BaseModel):
    """Text response configuration for OpenAI responses.

    :param format: (Optional) Text format configuration specifying output format requirements
    """

    format: OpenAIResponseTextFormat | None = None


# Must match type Literals of OpenAIResponseInputToolWebSearch below
WebSearchToolTypes = ["web_search", "web_search_preview", "web_search_preview_2025_03_11"]


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    """Web search tool configuration for OpenAI response inputs.

    :param type: Web search tool type variant to use
    :param search_context_size: (Optional) Size of search context, must be "low", "medium", or "high"
    """

    # Must match values of WebSearchToolTypes above
    type: Literal["web_search"] | Literal["web_search_preview"] | Literal["web_search_preview_2025_03_11"] = (
        "web_search"
    )
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(default="medium", pattern="^low|medium|high$")
    # TODO: add user_location


@json_schema_type
class OpenAIResponseInputToolFunction(BaseModel):
    """Function tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "function"
    :param name: Name of the function that can be called
    :param description: (Optional) Description of what the function does
    :param parameters: (Optional) JSON schema defining the function's parameters
    :param strict: (Optional) Whether to enforce strict parameter validation
    """

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None
    strict: bool | None = None


@json_schema_type
class OpenAIResponseInputToolFileSearch(BaseModel):
    """File search tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "file_search"
    :param vector_store_ids: List of vector store identifiers to search within
    :param filters: (Optional) Additional filters to apply to the search
    :param max_num_results: (Optional) Maximum number of search results to return (1-50)
    :param ranking_options: (Optional) Options for ranking and scoring search results
    """

    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int | None = Field(default=10, ge=1, le=50)
    ranking_options: FileSearchRankingOptions | None = None


class ApprovalFilter(BaseModel):
    """Filter configuration for MCP tool approval requirements.

    :param always: (Optional) List of tool names that always require approval
    :param never: (Optional) List of tool names that never require approval
    """

    always: list[str] | None = None
    never: list[str] | None = None


class AllowedToolsFilter(BaseModel):
    """Filter configuration for restricting which MCP tools can be used.

    :param tool_names: (Optional) List of specific tool names that are allowed
    """

    tool_names: list[str] | None = None


@json_schema_type
class OpenAIResponseInputToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param server_url: URL endpoint of the MCP server
    :param headers: (Optional) HTTP headers to include when connecting to the server
    :param require_approval: Approval requirement for tool calls ("always", "never", or filter)
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    headers: dict[str, Any] | None = None

    require_approval: Literal["always"] | Literal["never"] | ApprovalFilter = "never"
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseInputToolMCP,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


@json_schema_type
class OpenAIResponseToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response object.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = "mcp"
    server_label: str
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseToolMCP,  # The only type that differes from that in the inputs is the MCP tool
    Field(discriminator="type"),
]
register_schema(OpenAIResponseTool, name="OpenAIResponseTool")


class OpenAIResponseUsageOutputTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI response usage.

    :param reasoning_tokens: Number of tokens used for reasoning (o1/o3 models)
    """

    reasoning_tokens: int | None = None


class OpenAIResponseUsageInputTokensDetails(BaseModel):
    """Token details for input tokens in OpenAI response usage.

    :param cached_tokens: Number of tokens retrieved from cache
    """

    cached_tokens: int | None = None


@json_schema_type
class OpenAIResponseUsage(BaseModel):
    """Usage information for OpenAI response.

    :param input_tokens: Number of tokens in the input
    :param output_tokens: Number of tokens in the output
    :param total_tokens: Total tokens used (input + output)
    :param input_tokens_details: Detailed breakdown of input token usage
    :param output_tokens_details: Detailed breakdown of output token usage
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: OpenAIResponseUsageInputTokensDetails | None = None
    output_tokens_details: OpenAIResponseUsageOutputTokensDetails | None = None


@json_schema_type
class OpenAIResponseObject(BaseModel):
    """Complete OpenAI response object containing generation results and metadata.

    :param created_at: Unix timestamp when the response was created
    :param error: (Optional) Error details if the response generation failed
    :param id: Unique identifier for this response
    :param model: Model identifier used for generation
    :param object: Object type identifier, always "response"
    :param output: List of generated output items (messages, tool calls, etc.)
    :param parallel_tool_calls: Whether tool calls can be executed in parallel
    :param previous_response_id: (Optional) ID of the previous response in a conversation
    :param status: Current status of the response generation
    :param temperature: (Optional) Sampling temperature used for generation
    :param text: Text formatting configuration for the response
    :param top_p: (Optional) Nucleus sampling parameter used for generation
    :param tools: (Optional) An array of tools the model may call while generating a response.
    :param truncation: (Optional) Truncation strategy applied to the response
    :param usage: (Optional) Token usage information for the response
    """

    created_at: int
    error: OpenAIResponseError | None = None
    id: str
    model: str
    object: Literal["response"] = "response"
    output: list[OpenAIResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    status: str
    temperature: float | None = None
    # Default to text format to avoid breaking the loading of old responses
    # before the field was added. New responses will have this set always.
    text: OpenAIResponseText = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text"))
    top_p: float | None = None
    tools: list[OpenAIResponseTool] | None = None
    truncation: str | None = None
    usage: OpenAIResponseUsage | None = None


@json_schema_type
class OpenAIDeleteResponseObject(BaseModel):
    """Response object confirming deletion of an OpenAI response.

    :param id: Unique identifier of the deleted response
    :param object: Object type identifier, always "response"
    :param deleted: Deletion confirmation flag, always True
    """

    id: str
    object: Literal["response"] = "response"
    deleted: bool = True


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    """Streaming event indicating a new response has been created.

    :param response: The response object that was created
    :param type: Event type identifier, always "response.created"
    """

    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseObjectStreamResponseInProgress(BaseModel):
    """Streaming event indicating the response remains in progress.

    :param response: Current response state while in progress
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.in_progress"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.in_progress"] = "response.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    """Streaming event indicating a response has been completed.

    :param response: Completed response object
    :param type: Event type identifier, always "response.completed"
    """

    response: OpenAIResponseObject
    type: Literal["response.completed"] = "response.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseIncomplete(BaseModel):
    """Streaming event emitted when a response ends in an incomplete state.

    :param response: Response object describing the incomplete state
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.incomplete"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.incomplete"] = "response.incomplete"


@json_schema_type
class OpenAIResponseObjectStreamResponseFailed(BaseModel):
    """Streaming event emitted when a response fails.

    :param response: Response object describing the failure
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.failed"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.failed"] = "response.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemAdded(BaseModel):
    """Streaming event for when a new output item is added to the response.

    :param response_id: Unique identifier of the response containing this output
    :param item: The output item that was added (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.added"
    """

    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.added"] = "response.output_item.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemDone(BaseModel):
    """Streaming event for when an output item is completed.

    :param response_id: Unique identifier of the response containing this output
    :param item: The completed output item (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.done"
    """

    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.done"] = "response.output_item.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDelta(BaseModel):
    """Streaming event for incremental text content updates.

    :param content_index: Index position within the text content
    :param delta: Incremental text content being added
    :param item_id: Unique identifier of the output item being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.delta"
    """

    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.delta"] = "response.output_text.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDone(BaseModel):
    """Streaming event for when text output is completed.

    :param content_index: Index position within the text content
    :param text: Final complete text content of the output item
    :param item_id: Unique identifier of the completed output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.done"
    """

    content_index: int
    text: str  # final text of the output item
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.done"] = "response.output_text.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(BaseModel):
    """Streaming event for incremental function call argument updates.

    :param delta: Incremental function call arguments being added
    :param item_id: Unique identifier of the function call being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.delta"
    """

    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone(BaseModel):
    """Streaming event for when function call arguments are completed.

    :param arguments: Final complete arguments JSON string for the function call
    :param item_id: Unique identifier of the completed function call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.done"
    """

    arguments: str  # final arguments of the function call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallInProgress(BaseModel):
    """Streaming event for web search calls in progress.

    :param item_id: Unique identifier of the web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.in_progress"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.in_progress"] = "response.web_search_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallSearching(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.searching"] = "response.web_search_call.searching"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallCompleted(BaseModel):
    """Streaming event for completed web search calls.

    :param item_id: Unique identifier of the completed web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.completed"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.completed"] = "response.web_search_call.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsInProgress(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.in_progress"] = "response.mcp_list_tools.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsFailed(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.failed"] = "response.mcp_list_tools.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsCompleted(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.completed"] = "response.mcp_list_tools.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(BaseModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.delta"] = "response.mcp_call.arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDone(BaseModel):
    arguments: str  # final arguments of the MCP call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.done"] = "response.mcp_call.arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallInProgress(BaseModel):
    """Streaming event for MCP calls in progress.

    :param item_id: Unique identifier of the MCP call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.in_progress"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.in_progress"] = "response.mcp_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallFailed(BaseModel):
    """Streaming event for failed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.failed"
    """

    sequence_number: int
    type: Literal["response.mcp_call.failed"] = "response.mcp_call.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallCompleted(BaseModel):
    """Streaming event for completed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.completed"
    """

    sequence_number: int
    type: Literal["response.mcp_call.completed"] = "response.mcp_call.completed"


@json_schema_type
class OpenAIResponseContentPartOutputText(BaseModel):
    """Text content within a streamed response part.

    :param type: Content part type identifier, always "output_text"
    :param text: Text emitted for this content part
    :param annotations: Structured annotations associated with the text
    :param logprobs: (Optional) Token log probability details
    """

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = None


@json_schema_type
class OpenAIResponseContentPartReasoningText(BaseModel):
    """Reasoning text emitted as part of a streamed response.

    :param type: Content part type identifier, always "reasoning_text"
    :param text: Reasoning text supplied by the model
    """

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


OpenAIResponseContentPart = Annotated[
    OpenAIResponseContentPartOutputText | OpenAIResponseContentPartRefusal | OpenAIResponseContentPartReasoningText,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseContentPart, name="OpenAIResponseContentPart")


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartAdded(BaseModel):
    """Streaming event for when a new content part is added to a response item.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The content part that was added
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.added"
    """

    content_index: int
    response_id: str
    item_id: str
    output_index: int
    part: OpenAIResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.added"] = "response.content_part.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartDone(BaseModel):
    """Streaming event for when a content part is completed.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The completed content part
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.done"
    """

    content_index: int
    response_id: str
    item_id: str
    output_index: int
    part: OpenAIResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.done"] = "response.content_part.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningTextDelta(BaseModel):
    """Streaming event for incremental reasoning text updates.

    :param content_index: Index position of the reasoning content part
    :param delta: Incremental reasoning text being added
    :param item_id: Unique identifier of the output item being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.reasoning_text.delta"
    """

    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningTextDone(BaseModel):
    """Streaming event for when reasoning text is completed.

    :param content_index: Index position of the reasoning content part
    :param text: Final complete reasoning text
    :param item_id: Unique identifier of the completed output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.reasoning_text.done"
    """

    content_index: int
    text: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"


@json_schema_type
class OpenAIResponseContentPartReasoningSummary(BaseModel):
    """Reasoning summary part in a streamed response.

    :param type: Content part type identifier, always "summary_text"
    :param text: Summary text
    """

    type: Literal["summary_text"] = "summary_text"
    text: str


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded(BaseModel):
    """Streaming event for when a new reasoning summary part is added.

    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param part: The summary part that was added
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_part.added"
    """

    item_id: str
    output_index: int
    part: OpenAIResponseContentPartReasoningSummary
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_part.added"] = "response.reasoning_summary_part.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryPartDone(BaseModel):
    """Streaming event for when a reasoning summary part is completed.

    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param part: The completed summary part
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_part.done"
    """

    item_id: str
    output_index: int
    part: OpenAIResponseContentPartReasoningSummary
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_part.done"] = "response.reasoning_summary_part.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta(BaseModel):
    """Streaming event for incremental reasoning summary text updates.

    :param delta: Incremental summary text being added
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_text.delta"
    """

    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_text.delta"] = "response.reasoning_summary_text.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseReasoningSummaryTextDone(BaseModel):
    """Streaming event for when reasoning summary text is completed.

    :param text: Final complete summary text
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the output item
    :param sequence_number: Sequential number for ordering streaming events
    :param summary_index: Index of the summary part within the reasoning summary
    :param type: Event type identifier, always "response.reasoning_summary_text.done"
    """

    text: str
    item_id: str
    output_index: int
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_text.done"] = "response.reasoning_summary_text.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseRefusalDelta(BaseModel):
    """Streaming event for incremental refusal text updates.

    :param content_index: Index position of the content part
    :param delta: Incremental refusal text being added
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.refusal.delta"
    """

    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.refusal.delta"] = "response.refusal.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseRefusalDone(BaseModel):
    """Streaming event for when refusal text is completed.

    :param content_index: Index position of the content part
    :param refusal: Final complete refusal text
    :param item_id: Unique identifier of the output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.refusal.done"
    """

    content_index: int
    refusal: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.refusal.done"] = "response.refusal.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextAnnotationAdded(BaseModel):
    """Streaming event for when an annotation is added to output text.

    :param item_id: Unique identifier of the item to which the annotation is being added
    :param output_index: Index position of the output item in the response's output array
    :param content_index: Index position of the content part within the output item
    :param annotation_index: Index of the annotation within the content part
    :param annotation: The annotation object being added
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.annotation.added"
    """

    item_id: str
    output_index: int
    content_index: int
    annotation_index: int
    annotation: OpenAIResponseAnnotations
    sequence_number: int
    type: Literal["response.output_text.annotation.added"] = "response.output_text.annotation.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallInProgress(BaseModel):
    """Streaming event for file search calls in progress.

    :param item_id: Unique identifier of the file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.in_progress"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.file_search_call.in_progress"] = "response.file_search_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallSearching(BaseModel):
    """Streaming event for file search currently searching.

    :param item_id: Unique identifier of the file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.searching"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.file_search_call.searching"] = "response.file_search_call.searching"


@json_schema_type
class OpenAIResponseObjectStreamResponseFileSearchCallCompleted(BaseModel):
    """Streaming event for completed file search calls.

    :param item_id: Unique identifier of the completed file search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.file_search_call.completed"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.file_search_call.completed"] = "response.file_search_call.completed"


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated
    | OpenAIResponseObjectStreamResponseInProgress
    | OpenAIResponseObjectStreamResponseOutputItemAdded
    | OpenAIResponseObjectStreamResponseOutputItemDone
    | OpenAIResponseObjectStreamResponseOutputTextDelta
    | OpenAIResponseObjectStreamResponseOutputTextDone
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
    | OpenAIResponseObjectStreamResponseWebSearchCallInProgress
    | OpenAIResponseObjectStreamResponseWebSearchCallSearching
    | OpenAIResponseObjectStreamResponseWebSearchCallCompleted
    | OpenAIResponseObjectStreamResponseMcpListToolsInProgress
    | OpenAIResponseObjectStreamResponseMcpListToolsFailed
    | OpenAIResponseObjectStreamResponseMcpListToolsCompleted
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
    | OpenAIResponseObjectStreamResponseMcpCallInProgress
    | OpenAIResponseObjectStreamResponseMcpCallFailed
    | OpenAIResponseObjectStreamResponseMcpCallCompleted
    | OpenAIResponseObjectStreamResponseContentPartAdded
    | OpenAIResponseObjectStreamResponseContentPartDone
    | OpenAIResponseObjectStreamResponseReasoningTextDelta
    | OpenAIResponseObjectStreamResponseReasoningTextDone
    | OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded
    | OpenAIResponseObjectStreamResponseReasoningSummaryPartDone
    | OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta
    | OpenAIResponseObjectStreamResponseReasoningSummaryTextDone
    | OpenAIResponseObjectStreamResponseRefusalDelta
    | OpenAIResponseObjectStreamResponseRefusalDone
    | OpenAIResponseObjectStreamResponseOutputTextAnnotationAdded
    | OpenAIResponseObjectStreamResponseFileSearchCallInProgress
    | OpenAIResponseObjectStreamResponseFileSearchCallSearching
    | OpenAIResponseObjectStreamResponseFileSearchCallCompleted
    | OpenAIResponseObjectStreamResponseIncomplete
    | OpenAIResponseObjectStreamResponseFailed
    | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseInputFunctionToolCallOutput(BaseModel):
    """
    This represents the output of a function call that gets passed back to the model.
    """

    call_id: str
    output: str
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    status: str | None = None


OpenAIResponseInput = Annotated[
    # Responses API allows output messages to be passed in as input
    OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    | OpenAIResponseMCPApprovalRequest
    | OpenAIResponseMCPApprovalResponse
    |
    # Fallback to the generic message type as a last resort
    OpenAIResponseMessage,
    Field(union_mode="left_to_right"),
]
register_schema(OpenAIResponseInput, name="OpenAIResponseInput")


class ListOpenAIResponseInputItem(BaseModel):
    """List container for OpenAI response input items.

    :param data: List of input items
    :param object: Object type identifier, always "list"
    """

    data: list[OpenAIResponseInput]
    object: Literal["list"] = "list"


@json_schema_type
class OpenAIResponseObjectWithInput(OpenAIResponseObject):
    """OpenAI response object extended with input context information.

    :param input: List of input items that led to this response
    """

    input: list[OpenAIResponseInput]

    def to_response_object(self) -> OpenAIResponseObject:
        """Convert to OpenAIResponseObject by excluding input field."""
        return OpenAIResponseObject(**{k: v for k, v in self.model_dump().items() if k != "input"})


@json_schema_type
class ListOpenAIResponseObject(BaseModel):
    """Paginated list of OpenAI response objects with navigation metadata.

    :param data: List of response objects with their input context
    :param has_more: Whether there are more results available beyond this page
    :param first_id: Identifier of the first item in this page
    :param last_id: Identifier of the last item in this page
    :param object: Object type identifier, always "list"
    """

    data: list[OpenAIResponseObjectWithInput]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"
