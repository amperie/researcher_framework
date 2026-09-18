import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.platform.progress import stream_progress
from tests.test_platform_service import setup, headers, KEY_B
from tests.test_platform_workflow import request


def test_progress_arrives_while_model_is_still_running(postgres_dsn):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def invoke(_):
            started.set()
            await release.wait()
            return SimpleNamespace(content='{"content":"answer"}')
        service, _, _ = setup(postgres_dsn, invoke)
        task = asyncio.create_task(service.execute("a", request()))
        await asyncio.wait_for(started.wait(), 5)
        receipt = service.store.get_request("a", "r")
        stream = stream_progress(service.store, "a", "r", receipt)
        try:
            first = await asyncio.wait_for(anext(stream), 2)
            assert 'event: progress' in first and 'Preparing a reply' in first
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, 5)
            remaining = [frame async for frame in stream]
            assert '"status": "succeeded"' in remaining[-1]
            assert any('chat complete' in frame.lower() for frame in remaining)
        finally:
            release.set()
            await stream.aclose()
            await task
    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "stopped", "interrupted"])
def test_stream_replay_cursor_terminal_and_tenant_isolation(postgres_dsn, terminal):
    service, client, invoke = setup(postgres_dsn)
    store = service.store
    store.claim("a", "r", "s", "hash", 30)
    store.progress("a", "r", "research", "Retrieving articles")
    store.progress("a", "r", "research", "Retrieved article: <script>untrusted</script>")
    store.finish("a", "r", terminal)
    path = "/v1/requests/r/events"
    assert client.get(path).status_code == 401
    assert client.get(path, headers=headers(KEY_B)).status_code == 404
    assert client.get(path + "?after=-1", headers=headers()).status_code == 422
    response = client.get(path, headers=headers())
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.count('event: progress') == 2
    resumed = client.get(path, headers={**headers(), "Last-Event-ID": "1"}).text
    assert 'id: 1\n' not in resumed and 'id: 2\n' in resumed
    assert f'"status": "{terminal}"' in resumed
    assert client.get(path + "?after=2", headers=headers()).text.startswith('event: complete')
    invoke.assert_not_called()


def test_disconnecting_observer_does_not_stop_request(postgres_dsn):
    async def scenario():
        service, _, _ = setup(postgres_dsn)
        store = service.store
        store.claim("a", "r", "s", "hash", 30)
        store.progress("a", "r", "research", "Retrieving")
        stream = stream_progress(store, "a", "r", store.get_request("a", "r"))
        await anext(stream)
        await stream.aclose()
        assert store.get_request("a", "r")["status"] == "running"
        store.stop("a", "r")
    asyncio.run(scenario())


def test_browser_stream_reader_handles_split_utf8_frames_and_heartbeats():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to test the browser stream reader")
    script = Path("core/platform/activity.js").read_text(encoding="utf-8")
    script += r'''
const assert = require('node:assert/strict');
(async()=>{
  const expected = {sequence:1,message:'Retrieved café <script>literal</script>'};
  const wire = ': keep-alive\r\n\r\nid: 1\r\nevent: progress\r\ndata: '+JSON.stringify(expected)+'\r\n\r\nevent: complete\ndata: {"status":"succeeded"}\n\n';
  const bytes = new TextEncoder().encode(wire), events = [];
  const response = new Response(new ReadableStream({start(controller){for(const byte of bytes)controller.enqueue(Uint8Array.of(byte));controller.close()}}));
  await readActivityStream(response,(type,data)=>events.push([type,data]));
  assert.deepEqual(events,[['progress',expected],['complete',{status:'succeeded'}]]);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
