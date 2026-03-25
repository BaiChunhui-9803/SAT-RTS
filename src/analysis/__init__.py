"""
分析模块

包含模式分析和敏感性分析功能
"""

from .pattern import (
    read_csv,
    process_file,
    extract_continuous_patterns,
    replace_sequences_with_patterns,
    get_marked_sequences,
    handle_nested_patterns,
    build_pattern_dict,
    create_dataframe,
    plot_pattern_analysis,
)
from .pattern_batch import PatternAnalyzer
from .sensitivity import ParameterSensitivityAnalyzer

__all__ = [
    "PatternAnalyzer",
    "ParameterSensitivityAnalyzer",
    "read_csv",
    "process_file",
    "extract_continuous_patterns",
    "replace_sequences_with_patterns",
    "get_marked_sequences",
    "handle_nested_patterns",
    "build_pattern_dict",
    "create_dataframe",
    "plot_pattern_analysis",
]
