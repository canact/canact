# Vendored from Aider-AI/aider aider/models.py (Apache-2.0).
# Pinned 2026-09-03 from main. Only ModelSettings fields; no LiteLLM import.
# Used to prove canact export rows construct Aider's dataclass.

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ModelSettings:
    name: str
    edit_format: str = "whole"
    weak_model_name: Optional[str] = None
    use_repo_map: bool = False
    send_undo_reply: bool = False
    lazy: bool = False
    overeager: bool = False
    reminder: str = "user"
    examples_as_sys_msg: bool = False
    extra_params: Optional[dict] = None
    cache_control: bool = False
    caches_by_default: bool = False
    use_system_prompt: bool = True
    use_temperature: Union[bool, float] = True
    streaming: bool = True
    editor_model_name: Optional[str] = None
    editor_edit_format: Optional[str] = None
    reasoning_tag: Optional[str] = None
    remove_reasoning: Optional[str] = None
    system_prompt_prefix: Optional[str] = None
    accepts_settings: Optional[list] = None


def load_row(data: dict) -> ModelSettings:
    return ModelSettings(**data)


if __name__ == "__main__":
    row = json.load(sys.stdin)
    settings = load_row(row)
    if settings.edit_format not in ("diff", "udiff", "whole", "diff-fenced", "architect"):
        raise SystemExit(f"unexpected edit_format {settings.edit_format}")
    print(settings.name)
    print(settings.edit_format)
    print("ok")
