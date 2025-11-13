# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from collections.abc import Iterable
from dataclasses import dataclass

from llama_stack.log import get_logger

logger = get_logger(name=__name__)


@dataclass(frozen=True)
class RuntimeErrorRule:
    code: str
    default_message: str
    substrings: tuple[str, ...] = ()
    regex: re.Pattern[str] | None = None
    template: str | None = None

    def evaluate(self, error_msg: str) -> str | None:
        """
        Returns the sanitized message if the rule matches, otherwise None.
        """
        if self.regex:
            match = self.regex.search(error_msg)
            if match:
                if self.template:
                    try:
                        return self.template.format(**match.groupdict())
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("Failed to format sanitized runtime error message", exc_info=True)
                return self.default_message

        lowered = error_msg.lower()
        if self.substrings and all(pattern in lowered for pattern in self.substrings):
            return self.default_message

        return None


@dataclass(frozen=True)
class SanitizedRuntimeError:
    code: str
    message: str


MODEL_NOT_FOUND_REGEX = re.compile(r"model ['\"]?(?P<model>[^'\" ]+)['\"]? not found", re.IGNORECASE)


RUNTIME_ERROR_RULES: tuple[RuntimeErrorRule, ...] = (
    RuntimeErrorRule(
        code="MODEL_NOT_FOUND",
        default_message="Requested model is unavailable.",
        regex=MODEL_NOT_FOUND_REGEX,
        template="Requested model '{model}' is unavailable.",
    ),
)


def sanitize_runtime_error(
    error: RuntimeError, rules: Iterable[RuntimeErrorRule] = RUNTIME_ERROR_RULES
) -> SanitizedRuntimeError | None:
    """
    Map internal RuntimeError messages to stable, user-safe error codes/messages.
    """
    message = str(error)

    for rule in rules:
        sanitized_message = rule.evaluate(message)
        if sanitized_message:
            return SanitizedRuntimeError(code=rule.code, message=sanitized_message)

    return None
