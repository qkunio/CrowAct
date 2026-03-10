import json
from typing import Any, Iterator

import requests


def iter_sse_chunks(
    response: requests.Response, content_blocks: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    current_event = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue

        if not line.startswith("data:"):
            continue

        data_str = line[len("data:") :].strip()
        if data_str == "[DONE]":
            break

        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type") or current_event

        if event_type == "content_block_start":
            block = payload.get("content_block", {})
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                content_blocks.append({"type": "text", "text": text})
                if text:
                    yield {"type": "text", "text": text}

            elif block_type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": {},
                        "_partial_input": "",
                    }
                )

        elif event_type == "content_block_delta":
            index = payload.get("index")
            if index is None or index >= len(content_blocks):
                continue

            delta = payload.get("delta", {})
            delta_type = delta.get("type")
            block = content_blocks[index]

            if delta_type == "text_delta" and block.get("type") == "text":
                text = delta.get("text", "")
                block["text"] = block.get("text", "") + text
                if text:
                    yield {"type": "text", "text": text}

            elif delta_type == "input_json_delta" and block.get("type") == "tool_use":
                partial_json = delta.get("partial_json", "")
                block["_partial_input"] = block.get("_partial_input", "") + partial_json

        elif event_type == "content_block_stop":
            index = payload.get("index")
            if index is None or index >= len(content_blocks):
                continue

            block = content_blocks[index]
            if block.get("type") != "tool_use":
                continue

            partial_input = block.pop("_partial_input", "")
            if partial_input:
                try:
                    block["input"] = json.loads(partial_input)
                except json.JSONDecodeError:
                    block["input"] = {}

            yield {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input") or {},
            }

        elif event_type == "message_stop":
            break

        elif event_type == "error":
            raise RuntimeError(f"Streaming API returned an error: {payload}")
