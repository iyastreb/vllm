# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP bootstrap server for NIXL engine discovery.

Mirrors the Mooncake bootstrap design: a small server is launched on the
DP-master engine of each instance. Every engine registers its static identity
(engine_id, side-channel host/port, tp_size, connector mode) so that a router
can discover the NIXL coordinates and mode of an instance without parsing a
prefill response.
"""

import threading
import time

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vllm import envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

logger = init_logger(__name__)


class NixlEngineInfo(BaseModel):
    engine_id: str
    host: str
    port: int
    tp_size: int
    kv_connector: str


class RegisterEnginePayload(BaseModel):
    dp_rank: int
    info: NixlEngineInfo


def get_nixl_dp_engine_index(parallel_config: ParallelConfig) -> int:
    if parallel_config.local_engines_only:
        assert parallel_config.data_parallel_rank_local is not None
        return parallel_config.data_parallel_rank_local
    return parallel_config.data_parallel_index


def should_launch_nixl_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        return parallel_config.data_parallel_rank_local == 0
    return parallel_config.data_parallel_index == 0


def get_nixl_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    return host, envs.VLLM_NIXL_BOOTSTRAP_PORT


def register_engine_with_bootstrap(
    host: str, port: int, dp_rank: int, info: NixlEngineInfo
) -> None:
    url = make_zmq_path("http", host, port) + "/register"
    payload = RegisterEnginePayload(dp_rank=dp_rank, info=info)
    while True:
        try:
            response = requests.post(url, json=payload.model_dump(), timeout=5)
            response.raise_for_status()
            logger.debug("Registered NIXL engine with bootstrap server at %s", url)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
        except Exception as e:
            logger.error("Error registering NIXL engine with bootstrap: %s", e)
            return


class NixlBootstrapServer:
    """Server running on the DP-master engine that advertises engine identity."""

    def __init__(self, host: str, port: int):
        self.engines: dict[int, NixlEngineInfo] = {}

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        self.app.post("/register")(self.register_engine)
        self.app.get("/query", response_model=dict[int, NixlEngineInfo])(self.query)

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="nixl_bootstrap_server", daemon=True
        )
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)
        logger.info("NIXL Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("NIXL Bootstrap Server stopped.")

    async def register_engine(self, payload: RegisterEnginePayload):
        self.engines[payload.dp_rank] = payload.info
        logger.debug(
            "Registered NIXL engine: dp_rank=%d, engine_id=%s, %s:%d, mode=%s",
            payload.dp_rank,
            payload.info.engine_id,
            payload.info.host,
            payload.info.port,
            payload.info.kv_connector,
        )
        return {"status": "ok"}

    async def query(self) -> dict[int, NixlEngineInfo]:
        return self.engines
