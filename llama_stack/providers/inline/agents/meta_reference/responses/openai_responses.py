# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from collections.abc import AsyncIterator

from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseContentPartRefusal,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from llama_stack.apis.common.errors import (
    InvalidConversationIdError,
)
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.conversations.conversations import ConversationItem
from llama_stack.apis.inference import (
    Inference,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)

from .streaming import StreamingResponseOrchestrator
from .tool_executor import ToolExecutor
from .types import ChatCompletionContext, ToolContext
from .utils import (
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
    extract_shield_ids,
)

logger = get_logger(name=__name__, category="openai_responses")


class OpenAIResponsePreviousResponseWithInputItems(BaseModel):
    input_items: ListOpenAIResponseInputItem
    response: OpenAIResponseObject


class OpenAIResponsesImpl:
    def __init__(
        self,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        responses_store: ResponsesStore,
        vector_io_api: VectorIO,  # VectorIO
        safety_api: Safety,
        conversations_api: Conversations,
    ):
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.responses_store = responses_store
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.conversations_api = conversations_api
        self.tool_executor = ToolExecutor(
            tool_groups_api=tool_groups_api,
            tool_runtime_api=tool_runtime_api,
            vector_io_api=vector_io_api,
        )

    async def _prepend_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        previous_response: _OpenAIResponseObjectWithInputAndMessages,
    ):
        new_input_items = previous_response.input.copy()
        new_input_items.extend(previous_response.output)

        if isinstance(input, str):
            new_input_items.append(OpenAIResponseMessage(content=input, role="user"))
        else:
            new_input_items.extend(input)

        return new_input_items

    async def _process_input_with_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        tools: list[OpenAIResponseInputTool] | None,
        previous_response_id: str | None,
    ) -> tuple[str | list[OpenAIResponseInput], list[OpenAIMessageParam]]:
        """Process input with optional previous response context.

        Returns:
            tuple: (all_input for storage, messages for chat completion, tool context)
        """
        tool_context = ToolContext(tools)
        if previous_response_id:
            previous_response: _OpenAIResponseObjectWithInputAndMessages = (
                await self.responses_store.get_response_object(previous_response_id)
            )
            all_input = await self._prepend_previous_response(input, previous_response)

            if previous_response.messages:
                # Use stored messages directly and convert only new input
                message_adapter = TypeAdapter(list[OpenAIMessageParam])
                messages = message_adapter.validate_python(previous_response.messages)
                new_messages = await convert_response_input_to_chat_messages(input, previous_messages=messages)
                messages.extend(new_messages)
            else:
                # Backward compatibility: reconstruct from inputs
                messages = await convert_response_input_to_chat_messages(all_input)

            tool_context.recover_tools_from_previous_response(previous_response)
        else:
            all_input = input
            messages = await convert_response_input_to_chat_messages(input)

        return all_input, messages, tool_context

    async def _prepend_instructions(self, messages, instructions):
        if instructions:
            messages.insert(0, OpenAISystemMessageParam(content=instructions))

    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        response_with_input = await self.responses_store.get_response_object(response_id)
        return response_with_input.to_response_object()

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        return await self.responses_store.list_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items for a given OpenAI response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        :returns: An ListOpenAIResponseInputItem.
        """
        return await self.responses_store.list_response_input_items(response_id, after, before, include, limit, order)

    async def _store_response(
        self,
        response: OpenAIResponseObject,
        input: str | list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        new_input_id = f"msg_{uuid.uuid4()}"
        if isinstance(input, str):
            # synthesize a message from the input string
            input_content = OpenAIResponseInputMessageContentText(text=input)
            input_content_item = OpenAIResponseMessage(
                role="user",
                content=[input_content],
                id=new_input_id,
            )
            input_items_data = [input_content_item]
        else:
            # we already have a list of messages
            input_items_data = []
            for input_item in input:
                if isinstance(input_item, OpenAIResponseMessage):
                    # These may or may not already have an id, so dump to dict, check for id, and add if missing
                    input_item_dict = input_item.model_dump()
                    if "id" not in input_item_dict:
                        input_item_dict["id"] = new_input_id
                    input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                else:
                    input_items_data.append(input_item)

        await self.responses_store.store_response_object(
            response_object=response,
            input=input_items_data,
            messages=messages,
        )

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        shields: list | None = None,
    ):
        stream = bool(stream)
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")) if text is None else text

        shield_ids = extract_shield_ids(shields) if shields else []

        if conversation is not None and previous_response_id is not None:
            raise ValueError(
                "Mutually exclusive parameters: 'previous_response_id' and 'conversation'. Ensure you are only providing one of these parameters."
            )

        original_input = input  # needed for syncing to Conversations
        if conversation is not None:
            if not conversation.startswith("conv_"):
                raise InvalidConversationIdError(conversation)

            # Check conversation exists (raises ConversationNotFoundError if not)
            _ = await self.conversations_api.get_conversation(conversation)
            input = await self._load_conversation_context(conversation, input)

        stream_gen = self._create_streaming_response(
            input=input,
            original_input=original_input,
            model=model,
            instructions=instructions,
            previous_response_id=previous_response_id,
            conversation=conversation,
            store=store,
            temperature=temperature,
            text=text,
            tools=tools,
            max_infer_iters=max_infer_iters,
            shield_ids=shield_ids,
        )

        if stream:
            return stream_gen
        else:
            final_response = None
            final_event_type = None
            failed_response = None

            async for stream_chunk in stream_gen:
                if stream_chunk.type in {"response.completed", "response.incomplete"}:
                    if final_response is not None:
                        raise ValueError(
                            "The response stream produced multiple terminal responses! "
                            f"Earlier response from {final_event_type}"
                        )
                    final_response = stream_chunk.response
                    final_event_type = stream_chunk.type
                elif stream_chunk.type == "response.failed":
                    failed_response = stream_chunk.response

            if failed_response is not None:
                error_message = (
                    failed_response.error.message
                    if failed_response and failed_response.error
                    else "Response stream failed without error details"
                )
                raise RuntimeError(f"OpenAI response failed: {error_message}")

            if final_response is None:
                raise ValueError("The response stream never reached a terminal state")
            return final_response

    async def _create_refusal_response_events(
        self, refusal_content: OpenAIResponseContentPartRefusal, response_id: str, created_at: int, model: str
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Create and yield refusal response events following the established streaming pattern."""
        # Create initial response and yield created event
        initial_response = OpenAIResponseObject(
            id=response_id,
            created_at=created_at,
            model=model,
            status="in_progress",
            output=[],
        )
        yield OpenAIResponseObjectStreamResponseCreated(response=initial_response)

        # Create completed refusal response using OpenAIResponseContentPartRefusal
        refusal_response = OpenAIResponseObject(
            id=response_id,
            created_at=created_at,
            model=model,
            status="completed",
            output=[OpenAIResponseMessage(role="assistant", content=[refusal_content], type="message")],
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=refusal_response)

    async def _create_streaming_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        original_input: str | list[OpenAIResponseInput] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        max_infer_iters: int | None = 10,
        shield_ids: list[str] | None = None,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Input preprocessing
        all_input, messages, tool_context = await self._process_input_with_previous_response(
            input, tools, previous_response_id
        )
        await self._prepend_instructions(messages, instructions)

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            temperature=temperature,
            response_format=response_format,
            tool_context=tool_context,
            inputs=all_input,
        )

        # Create orchestrator and delegate streaming logic
        response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        orchestrator = StreamingResponseOrchestrator(
            inference_api=self.inference_api,
            ctx=ctx,
            response_id=response_id,
            created_at=created_at,
            text=text,
            max_infer_iters=max_infer_iters,
            tool_executor=self.tool_executor,
            safety_api=self.safety_api,
            shield_ids=shield_ids,
        )

        # Output safety validation hook - delegated to streaming orchestrator for real-time validation
        # Stream the response
        final_response = None
        failed_response = None
        async for stream_chunk in orchestrator.create_response():
            if stream_chunk.type in {"response.completed", "response.incomplete"}:
                final_response = stream_chunk.response
            elif stream_chunk.type == "response.failed":
                failed_response = stream_chunk.response
            yield stream_chunk

            # Store and sync immediately after yielding terminal events
            # This ensures the storage/syncing happens even if the consumer breaks early
            if (
                stream_chunk.type in {"response.completed", "response.incomplete"}
                and store
                and final_response
                and failed_response is None
            ):
                await self._store_response(
                    response=final_response,
                    input=all_input,
                    messages=orchestrator.final_messages,
                )

            if stream_chunk.type in {"response.completed", "response.incomplete"} and conversation and final_response:
                # for Conversations, we need to use the original_input if it's available, otherwise use input
                sync_input = original_input if original_input is not None else input
                await self._sync_response_to_conversation(conversation, sync_input, final_response)

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        return await self.responses_store.delete_response_object(response_id)

    async def _load_conversation_context(
        self, conversation_id: str, content: str | list[OpenAIResponseInput]
    ) -> list[OpenAIResponseInput]:
        """Load conversation history and merge with provided content."""
        conversation_items = await self.conversations_api.list(conversation_id, order="asc")

        context_messages = []
        for item in conversation_items.data:
            if isinstance(item, OpenAIResponseMessage):
                if item.role == "user":
                    context_messages.append(
                        OpenAIResponseMessage(
                            role="user", content=item.content, id=item.id if hasattr(item, "id") else None
                        )
                    )
                elif item.role == "assistant":
                    context_messages.append(
                        OpenAIResponseMessage(
                            role="assistant", content=item.content, id=item.id if hasattr(item, "id") else None
                        )
                    )

        # add new content to context
        if isinstance(content, str):
            context_messages.append(OpenAIResponseMessage(role="user", content=content))
        elif isinstance(content, list):
            context_messages.extend(content)

        return context_messages

    async def _sync_response_to_conversation(
        self, conversation_id: str, content: str | list[OpenAIResponseInput], response: OpenAIResponseObject
    ) -> None:
        """Sync content and response messages to the conversation."""
        conversation_items = []

        # add user content message(s)
        if isinstance(content, str):
            conversation_items.append(
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]}
            )
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, OpenAIResponseMessage):
                    raise NotImplementedError(f"Unsupported input item type: {type(item)}")

                if item.role == "user":
                    if isinstance(item.content, str):
                        conversation_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": item.content}],
                            }
                        )
                    elif isinstance(item.content, list):
                        conversation_items.append({"type": "message", "role": "user", "content": item.content})
                    else:
                        raise NotImplementedError(f"Unsupported user message content type: {type(item.content)}")
                elif item.role == "assistant":
                    if isinstance(item.content, list):
                        conversation_items.append({"type": "message", "role": "assistant", "content": item.content})
                    else:
                        raise NotImplementedError(f"Unsupported assistant message content type: {type(item.content)}")
                else:
                    raise NotImplementedError(f"Unsupported message role: {item.role}")

        # add assistant response message
        for output_item in response.output:
            if isinstance(output_item, OpenAIResponseMessage) and output_item.role == "assistant":
                if hasattr(output_item, "content") and isinstance(output_item.content, list):
                    conversation_items.append({"type": "message", "role": "assistant", "content": output_item.content})

        if conversation_items:
            adapter = TypeAdapter(list[ConversationItem])
            validated_items = adapter.validate_python(conversation_items)
            await self.conversations_api.add_items(conversation_id, validated_items)
