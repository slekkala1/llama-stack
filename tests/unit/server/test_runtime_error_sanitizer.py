# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.server.runtime_error_sanitizer import sanitize_runtime_error


def test_model_not_found_is_sanitized():
    err = RuntimeError("OpenAI response failed: Model 'claude-sonnet-4-5-20250929' not found.")

    sanitized = sanitize_runtime_error(err)

    assert sanitized.code == "MODEL_NOT_FOUND"
    assert sanitized.message == "Requested model 'claude-sonnet-4-5-20250929' is unavailable."


def test_unmapped_runtime_error_defaults_to_internal_error():
    err = RuntimeError("Unexpected failure in obscure subsystem")

    sanitized = sanitize_runtime_error(err)

    assert sanitized is None
