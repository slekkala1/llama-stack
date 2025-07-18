# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from llama_stack_cli import LlamaStackCli, AsyncLlamaStackCli
from llama_stack_cli.types import StoreListInventoryResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestStore:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip()
    @parametrize
    def test_method_list_inventory(self, client: LlamaStackCli) -> None:
        store = client.store.list_inventory()
        assert_matches_type(StoreListInventoryResponse, store, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_raw_response_list_inventory(self, client: LlamaStackCli) -> None:
        response = client.store.with_raw_response.list_inventory()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        store = response.parse()
        assert_matches_type(StoreListInventoryResponse, store, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_streaming_response_list_inventory(self, client: LlamaStackCli) -> None:
        with client.store.with_streaming_response.list_inventory() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            store = response.parse()
            assert_matches_type(StoreListInventoryResponse, store, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncStore:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip()
    @parametrize
    async def test_method_list_inventory(self, async_client: AsyncLlamaStackCli) -> None:
        store = await async_client.store.list_inventory()
        assert_matches_type(StoreListInventoryResponse, store, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_raw_response_list_inventory(self, async_client: AsyncLlamaStackCli) -> None:
        response = await async_client.store.with_raw_response.list_inventory()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        store = await response.parse()
        assert_matches_type(StoreListInventoryResponse, store, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_streaming_response_list_inventory(self, async_client: AsyncLlamaStackCli) -> None:
        async with async_client.store.with_streaming_response.list_inventory() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            store = await response.parse()
            assert_matches_type(StoreListInventoryResponse, store, path=["response"])

        assert cast(Any, response.is_closed) is True
