"""Replayable status stream backed by tenant-scoped request receipts."""
import asyncio
import json


async def stream_progress(store, tenant, request_id, receipt, after=0):
    def frame(event, data, sequence=None):
        prefix = f"id: {sequence}\n" if sequence is not None else ""
        return prefix + f"event: {event}\ndata: {json.dumps(data)}\n\n"

    heartbeat = asyncio.get_running_loop().time()
    while True:
        for event in receipt["progress"]:
            if event["sequence"] > after:
                yield frame("progress", {"tenantId": tenant, "requestId": request_id, **event}, event["sequence"])
                after = event["sequence"]
        if receipt["status"] != "running":
            yield frame("complete", {key: receipt[key] for key in ("tenantId", "requestId", "status", "stage")})
            return
        now = asyncio.get_running_loop().time()
        if now - heartbeat >= 15:
            yield ": keep-alive\n\n"
            heartbeat = now
        await asyncio.sleep(0.5)
        receipt = await asyncio.to_thread(store.get_request, tenant, request_id)
