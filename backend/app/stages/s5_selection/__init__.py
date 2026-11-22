"""⑤ 资产选择（对齐 old-version asset_selection.md）：规则引擎 + 可选 LLM 复核。"""
from .agent import SelectionAgent
from .rules import DimRef, SelectionInput, rule_select
from .stage import SelectionStage

__all__ = ["DimRef", "SelectionAgent", "SelectionInput", "SelectionStage", "rule_select"]
