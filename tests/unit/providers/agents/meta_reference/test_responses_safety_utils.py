# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.agents.agents import ResponseShieldSpec
from llama_stack.apis.inference import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    convert_openai_to_inference_messages,
    extract_shield_ids,
    extract_text_content,
)


@pytest.fixture
def mock_apis():
    """Create mock APIs for testing."""
    return {
        "inference_api": AsyncMock(),
        "tool_groups_api": AsyncMock(),
        "tool_runtime_api": AsyncMock(),
        "responses_store": AsyncMock(),
        "vector_io_api": AsyncMock(),
        "safety_api": AsyncMock(),
    }


@pytest.fixture
def responses_impl(mock_apis):
    """Create OpenAIResponsesImpl instance with mocked dependencies."""
    return OpenAIResponsesImpl(**mock_apis)


def test_extract_shield_ids_from_strings(responses_impl):
    """Test extraction from simple string shield IDs."""
    shields = ["llama-guard", "content-filter", "nsfw-detector"]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_shield_ids_from_objects(responses_impl):
    """Test extraction from ResponseShieldSpec objects."""
    shields = [
        ResponseShieldSpec(type="llama-guard"),
        ResponseShieldSpec(type="content-filter"),
    ]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter"]


def test_extract_shield_ids_mixed_formats(responses_impl):
    """Test extraction from mixed string and object formats."""
    shields = [
        "llama-guard",
        ResponseShieldSpec(type="content-filter"),
        "nsfw-detector",
    ]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_shield_ids_none_input(responses_impl):
    """Test extraction with None input."""
    result = extract_shield_ids(None)
    assert result == []


def test_extract_shield_ids_empty_list(responses_impl):
    """Test extraction with empty list."""
    result = extract_shield_ids([])
    assert result == []


def test_extract_shield_ids_unknown_format(responses_impl, caplog):
    """Test extraction with unknown shield format logs warning."""
    # Create an object that's neither string nor ResponseShieldSpec
    unknown_object = {"invalid": "format"}  # Plain dict, not ResponseShieldSpec
    shields = ["valid-shield", unknown_object, "another-shield"]
    result = extract_shield_ids(shields)
    assert result == ["valid-shield", "another-shield"]
    assert "Unknown shield format" in caplog.text


def test_extract_text_content_string(responses_impl):
    """Test extraction from simple string content."""
    content = "Hello world"
    result = extract_text_content(content)
    assert result == "Hello world"


def test_extract_text_content_list_with_text(responses_impl):
    """Test extraction from list content with text parts."""
    content = [
        MagicMock(text="Hello "),
        MagicMock(text="world"),
    ]
    result = extract_text_content(content)
    assert result == "Hello  world"


def test_extract_text_content_list_with_refusal(responses_impl):
    """Test extraction skips refusal parts."""
    # Create text parts
    text_part1 = MagicMock()
    text_part1.text = "Hello"

    text_part2 = MagicMock()
    text_part2.text = "world"

    # Create refusal part (no text attribute)
    refusal_part = MagicMock()
    refusal_part.type = "refusal"
    refusal_part.refusal = "Blocked"
    del refusal_part.text  # Remove text attribute

    content = [text_part1, refusal_part, text_part2]
    result = extract_text_content(content)
    assert result == "Hello world"


def test_extract_text_content_empty_list(responses_impl):
    """Test extraction from empty list returns None."""
    content = []
    result = extract_text_content(content)
    assert result is None


def test_extract_text_content_no_text_parts(responses_impl):
    """Test extraction with no text parts returns None."""
    # Create image part (no text attribute)
    image_part = MagicMock()
    image_part.type = "image"
    image_part.image_url = "http://example.com"

    # Create refusal part (no text attribute)
    refusal_part = MagicMock()
    refusal_part.type = "refusal"
    refusal_part.refusal = "Blocked"

    # Explicitly remove text attributes to simulate non-text parts
    if hasattr(image_part, "text"):
        delattr(image_part, "text")
    if hasattr(refusal_part, "text"):
        delattr(refusal_part, "text")

    content = [image_part, refusal_part]
    result = extract_text_content(content)
    assert result is None


def test_extract_text_content_none_input(responses_impl):
    """Test extraction with None input returns None."""
    result = extract_text_content(None)
    assert result is None


def test_convert_user_message(responses_impl):
    """Test conversion of user message."""
    openai_msg = MagicMock(role="user", content="Hello world")
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Hello world"


def test_convert_system_message(responses_impl):
    """Test conversion of system message."""
    openai_msg = MagicMock(role="system", content="You are helpful")
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are helpful"


def test_convert_assistant_message(responses_impl):
    """Test conversion of assistant message."""
    openai_msg = MagicMock(role="assistant", content="I can help")
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 1
    assert isinstance(result[0], CompletionMessage)
    assert result[0].content == "I can help"
    assert result[0].stop_reason == StopReason.end_of_turn


def test_convert_tool_message_skipped(responses_impl):
    """Test that tool messages are skipped."""
    openai_msg = MagicMock(role="tool", content="Tool result")
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 0


def test_convert_complex_content(responses_impl):
    """Test conversion with complex content structure."""
    openai_msg = MagicMock(
        role="user",
        content=[
            MagicMock(text="Analyze this: "),
            MagicMock(text="important content"),
        ],
    )
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Analyze this:  important content"


def test_convert_empty_content_skipped(responses_impl):
    """Test that messages with no extractable content are skipped."""
    openai_msg = MagicMock(role="user", content=[])
    result = convert_openai_to_inference_messages([openai_msg])

    assert len(result) == 0


def test_convert_assistant_message_dict_format(responses_impl):
    """Test conversion of assistant message in dictionary format."""
    dict_msg = {"role": "assistant", "content": "Violent content refers to media, materials, or expressions"}
    result = convert_openai_to_inference_messages([dict_msg])

    assert len(result) == 1
    assert isinstance(result[0], CompletionMessage)
    assert result[0].content == "Violent content refers to media, materials, or expressions"
    assert result[0].stop_reason == StopReason.end_of_turn
