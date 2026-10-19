"""Decode only the top-level answer string from an unfinished JSON response."""
from __future__ import annotations

import json
import re


class AnswerTextDecoder:
    def __init__(self) -> None:
        self.content = ""
        self.emitted = ""

    def feed(self, delta: str) -> str:
        self.content += delta
        if len(self.content) > 131072:
            raise ValueError("streamed answer exceeded the response limit")
        start = self._answer_start()
        if start is None:
            return ""
        index = start
        boundary = start
        while index < len(self.content):
            char = self.content[index]
            if char == '"':
                break
            if char == "\\":
                if index + 1 >= len(self.content):
                    break
                width = 6 if self.content[index + 1] == "u" else 2
                if index + width > len(self.content):
                    break
                index += width
            else:
                index += 1
            boundary = index
        decoded = json.loads('"' + self.content[start:boundary] + '"')
        if decoded and 0xD800 <= ord(decoded[-1]) <= 0xDBFF:
            decoded = decoded[:-1]
        if not decoded.startswith(self.emitted):
            raise ValueError("answer stream changed previously delivered text")
        new = decoded[len(self.emitted):]
        self.emitted = decoded
        return new

    def _answer_start(self) -> int | None:
        index = self.content.find("{")
        if index < 0:
            return None
        index += 1
        decoder = json.JSONDecoder()
        while index < len(self.content):
            match = re.match(r"[\s,]*", self.content[index:])
            index += match.end()
            try:
                key, end = decoder.raw_decode(self.content, index)
            except json.JSONDecodeError:
                return None
            if not isinstance(key, str):
                return None
            index = end
            match = re.match(r'\s*:\s*', self.content[index:])
            if not match:
                return None
            index += match.end()
            if key == "answer":
                return index + 1 if self.content[index:index + 1] == '"' else None
            try:
                _, index = decoder.raw_decode(self.content, index)
            except json.JSONDecodeError:
                return None
