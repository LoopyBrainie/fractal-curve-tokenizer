# -*- coding: utf-8 -*-
"""Vectorization Audit Tool for FractalCurveViT

This module provides comprehensive tools for detecting non-vectorized code
in model architecture and its components. It includes:

1. Static AST Analysis: Detect Python for/while loops and recursion in forward paths
2. Dynamic FX Tracing: Use torch.fx.symbolic_trace to identify tracing failures
3. Profiler-based Detection: Monitor aten::item and aten::select call patterns
4. Recursion Depth Audit: Track recursion depth vs execution time ratios

Usage:
    # As decorator
    @audit_vectorization
    def my_forward_function(x):
        ...

    # As context manager
    with vectorization_audit("my_component"):
        ...

    # Direct inspection
    report = analyze_module_for_vectorization(MyModule)
"""

from __future__ import annotations

import ast
import inspect
import time
import warnings
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.fx
from torch import Tensor

# =============================================================================
# Configuration and Global State
# =============================================================================

# Global debug flag - when False, all audit operations are no-ops
_VECTORIZATION_AUDIT_ENABLED: bool = False

# Threshold for aten::item / aten::select calls per inference
_ITEM_SELECT_THRESHOLD: int = 10

# Threshold for scalar iteration detection (calls per batch * scales)
_SCALAR_ITER_THRESHOLD_FACTOR: float = 1.0


def enable_vectorization_audit(enabled: bool = True) -> None:
    """Enable or disable vectorization auditing globally."""
    global _VECTORIZATION_AUDIT_ENABLED
    _VECTORIZATION_AUDIT_ENABLED = enabled


def is_audit_enabled() -> bool:
    """Check if vectorization auditing is enabled."""
    return _VECTORIZATION_AUDIT_ENABLED


# =============================================================================
# Data Classes for Audit Results
# =============================================================================


@dataclass
class VectorizationIssue:
    """Represents a detected vectorization issue."""
    issue_type: str  # 'loop', 'recursion', 'fx_tracing_failure', 'scalar_iteration'
    location: Tuple[str, int]  # (file_path, line_number)
    function_name: str
    description: str
    severity: str = "warning"  # 'warning', 'error', 'performance'
    suggestion: str = ""
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizationReport:
    """Comprehensive report of vectorization analysis."""
    module_name: str
    issues: List[VectorizationIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    analysis_time: float = 0.0

    def add_issue(self, issue: VectorizationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        self.summary[issue.issue_type] = self.summary.get(issue.issue_type, 0) + 1

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.issues) > 0

    def get_performance_warnings(self) -> List[VectorizationIssue]:
        """Get all performance-related warnings."""
        return [i for i in self.issues if i.severity == "performance"]

    def get_errors(self) -> List[VectorizationIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == "error"]

    def __str__(self) -> str:
        lines = [f"Vectorization Report: {self.module_name}"]
        lines.append(f"Analysis time: {self.analysis_time:.3f}s")
        lines.append(f"Issues found: {len(self.issues)}")
        for issue_type, count in self.summary.items():
            lines.append(f"  - {issue_type}: {count}")
        if self.issues:
            lines.append("\nDetails:")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.function_name}:{issue.location[1]}")
                lines.append(f"    {issue.description}")
        return "\n".join(lines)


# =============================================================================
# Static AST Analysis
# =============================================================================


class ASTVectorizationAnalyzer:
    """AST-based analyzer for detecting non-vectorized patterns."""

    # Methods that are considered forward paths
    FORWARD_METHODS = {"forward", "__call__"}

    # Methods that are NOT forward paths (init, data loading, etc.)
    IGNORED_METHODS = {
        "__init__", "__new__", "__init_subclass__", "__class_getitem__",
        "_load_data", "__getitem__", "__len__", "__iter__", "__next__",
        "dataclass_fields", "_get_data", "load", "save", "collate_fn",
    }

    # Loop-related AST node types
    LOOP_NODES = (ast.For, ast.While, ast.AsyncFor)

    # Known vectorized PyTorch functions (safe to ignore)
    VECTORIZED_FUNCTIONS = {
        "torch.no_grad", "torch.enable_grad", "torch.inference_mode",
        "torch.arange", "torch.linspace", "torch.eye", "torch.zeros",
        "torch.ones", "torch.full", "torch.empty", "torch.rand",
        "torch.randn", "torch.randint", "torch.normal", "torch.linspace",
        "torch.meshgrid", "torch.stack", "torch.cat", "torch.concat",
        "torch.sum", "torch.mean", "torch.std", "torch.var",
        "torch.min", "torch.max", "torch.amin", "torch.amax",
        "torch.einsum", "torch.gather", "torch.scatter",
        "torch.topk", "torch.sort", "torch.argsort", "torch.where",
    }

    def __init__(self, ignore_init: bool = True):
        self.ignore_init = ignore_init
        self.issues: List[VectorizationIssue] = []
        self._source_code: Optional[str] = None  # P2-3 修复: 存储 source_code 用于 AST 解析

    def analyze_module(self, module: Type, source_path: Optional[str] = None) -> List[VectorizationIssue]:
        """Analyze a module class for vectorization issues."""
        self.issues = []
        self._source_code = None  # P2-3 修复: 重置 source_code

        # Get source code
        if source_path is None:
            source_path = self._get_source_path(module)

        if source_path is None or not Path(source_path).exists():
            return self.issues

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                self._source_code = f.read()  # P2-3 修复: 存储 source_code

            tree = ast.parse(self._source_code)
            self._analyze_tree(tree, module.__name__, source_path)

        except (SyntaxError, OSError, UnicodeDecodeError) as e:
            warnings.warn(f"Could not parse {source_path}: {e}")

        return self.issues

    def _get_source_path(self, obj: Any) -> Optional[str]:
        """Get the source file path for an object."""
        try:
            return inspect.getfile(obj)
        except TypeError:
            return None

    def _analyze_tree(
        self,
        tree: ast.AST,
        module_name: str,
        file_path: str,
        in_forward_method: bool = False,
        current_function: Optional[str] = None,
        source_code: Optional[str] = None,
    ) -> None:
        """Recursively analyze AST tree for issues."""
        # P2-3 修复: 使用传入的 source_code 或实例变量
        source_code = source_code or self._source_code

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_name = node.name
                is_forward = func_name in self.FORWARD_METHODS
                is_ignored = func_name in self.IGNORED_METHODS

                # Track if we're inside a forward method
                was_in_forward = in_forward_method
                in_forward_method = is_forward
                current_function = func_name

                # Only report issues in forward paths, not in __init__ or ignored methods
                should_analyze = in_forward_method and not is_ignored

                # Analyze function body for loops
                for child in ast.walk(node):
                    if isinstance(child, self.LOOP_NODES) and should_analyze:
                        self._check_loop(node, child, file_path, module_name, current_function, source_code)

                # Check for recursion
                if should_analyze:
                    self._check_recursion(node, file_path, module_name, current_function)

                # Recursively analyze nested functions (but only if in forward path)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._analyze_tree(
                            ast.parse(""),
                            module_name,
                            file_path,
                            in_forward_method=True,
                            current_function=f"{current_function}.{child.name}",
                            source_code=source_code,
                        )

                # Restore previous state
                in_forward_method = was_in_forward

    def _check_loop(
        self,
        func_node: ast.FunctionDef,
        loop_node: ast.AST,
        file_path: str,
        module_name: str,
        func_name: str,
        source_code: Optional[str] = None,
    ) -> None:
        """Check if a loop represents a potential vectorization issue."""
        loop_type = "for" if isinstance(loop_node, ast.For) else "while"

        # P2-3 修复: 使用 source_code 而非 inspect.getsource(func_node)
        # ast.get_source_segment 需要源代码字符串而非 AST 节点
        source_code = source_code or self._source_code
        if source_code:
            loop_code = ast.get_source_segment(source_code, loop_node) or ""
        else:
            loop_code = ""

        # Skip known vectorized patterns
        if any(pattern in loop_code for pattern in ["torch.", "for ", "while "]):
            # Check for common non-vectorized patterns
            non_vectorized_patterns = [
                ".item()",  # Scalar extraction
                ".item(",   # Another form
                "[i]",     # Index iteration over batch
                ".append(", # List building
                ".extend(", # List extension
                "range(len(",  # Python loop over tensor
            ]

            if any(pattern in loop_code for pattern in non_vectorized_patterns):
                self.issues.append(
                    VectorizationIssue(
                        issue_type="loop",
                        location=(file_path, loop_node.lineno),
                        function_name=func_name,
                        description=f"Potential non-vectorized {loop_type} loop detected",
                        severity="performance",
                        suggestion="Consider using torch operations instead of Python loops",
                        extra_info={"loop_type": loop_type, "code_snippet": loop_code[:100]},
                    )
                )

    def _check_recursion(
        self,
        func_node: ast.FunctionDef,
        file_path: str,
        module_name: str,
        func_name: str,
    ) -> None:
        """Check for recursive calls within a function."""
        # Find function calls that look like recursion
        # by walking through the AST body looking for Call nodes
        func_name_short = func_name.split(".")[-1]

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Check for direct function name call
                if isinstance(node.func, ast.Name) and node.func.id == func_name_short:
                    self.issues.append(
                        VectorizationIssue(
                            issue_type="recursion",
                            location=(file_path, func_node.lineno),
                            function_name=func_name,
                            description="Recursive call detected in forward path",
                            severity="warning",
                            suggestion="Recursion in forward pass may cause performance issues. Consider iterative implementation.",
                        )
                    )
                    return
                # Check for self.method call
                if isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name == func_name_short:
                        self.issues.append(
                            VectorizationIssue(
                                issue_type="recursion",
                                location=(file_path, func_node.lineno),
                                function_name=func_name,
                                description="Recursive call detected in forward path",
                                severity="warning",
                                suggestion="Recursion in forward pass may cause performance issues. Consider iterative implementation.",
                            )
                        )
                        return


# =============================================================================
# Dynamic FX Tracing Check
# =============================================================================


class FXTracingAnalyzer:
    """Analyzer using torch.fx.symbolic_trace to detect non-vectorized patterns."""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.issues: List[VectorizationIssue] = []

    def analyze_forward(
        self,
        model: torch.nn.Module,
        sample_input: Tensor,
        module_name: str = "unknown",
    ) -> List[VectorizationIssue]:
        """Try to trace the model's forward pass and detect issues."""
        self.issues = []

        try:
            # Attempt symbolic trace
            traced = torch.fx.symbolic_trace(model, sample_args=(sample_input,))

            # If successful, check graph structure
            self._analyze_graph(traced, module_name)

        except torch.fx.proxy.TraceError as e:
            self.issues.append(
                VectorizationIssue(
                    issue_type="fx_tracing_failure",
                    location=(module_name, 0),
                    function_name="forward",
                    description="FX symbolic_trace failed - likely contains data-dependent control flow",
                    severity="warning",
                    suggestion="Data-dependent control flow prevents vectorization. Consider using static control flow.",
                    extra_info={"error": str(e)},
                )
            )

        except Exception:
            # Other errors (not necessarily vectorization issues)
            pass

        return self.issues

    def _analyze_graph(self, traced: torch.fx.GraphModule, module_name: str) -> None:
        """Analyze the traced graph for potential issues."""
        graph = traced.graph

        # Count operations that suggest scalar iteration
        item_count = 0
        select_count = 0
        loop_ops = []

        for node in graph.nodes:
            # Check for aten::item (scalar extraction)
            if "item" in str(node.target):
                item_count += 1
                if item_count > self.batch_size * _ITEM_SELECT_THRESHOLD:
                    self.issues.append(
                        VectorizationIssue(
                            issue_type="scalar_iteration",
                            location=(module_name, 0),
                            function_name="forward",
                            description=f"High number of aten::item calls ({item_count}) detected",
                            severity="performance",
                            suggestion="aten::item() breaks vectorization. Consider keeping values as tensors.",
                        )
                    )
                    break

            # Check for aten::select (tensor indexing that may indicate scalar iteration)
            if "select" in str(node.target):
                select_count += 1

            # Check for control flow nodes
            if node.op in ("call_function",):
                target = str(node.target)
                if "loop" in target.lower() or "while" in target.lower():
                    loop_ops.append(node)

        if loop_ops:
            self.issues.append(
                VectorizationIssue(
                    issue_type="loop",
                    location=(module_name, 0),
                    function_name="forward",
                    description=f"Loop operations detected in traced graph: {len(loop_ops)}",
                    severity="performance",
                    suggestion="Consider if loops can be replaced with tensor operations.",
                )
            )


# =============================================================================
# Profiler-based Scalar Iteration Detection
# =============================================================================


class ProfilerScalarIteratorDetector:
    """Use torch.profiler to detect scalar iteration patterns."""

    def __init__(self, item_threshold: int = 10, select_threshold: int = 100):
        self.item_threshold = item_threshold
        self.select_threshold = select_threshold
        self.issues: List[VectorizationIssue] = []

    def analyze_inference(
        self,
        model: torch.nn.Module,
        input_tensor: Tensor,
        module_name: str = "unknown",
        batch_size: Optional[int] = None,
    ) -> List[VectorizationIssue]:
        """Run inference with profiler to detect scalar operations."""
        self.issues = []
        batch_size = batch_size or input_tensor.shape[0]

        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                    if torch.cuda.is_available()
                    else torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                profile_memory=False,
            ) as prof:
                # Run inference
                with torch.no_grad():
                    model(input_tensor)

            # Analyze profiler results
            events = prof.events()

            # Count aten::item and aten::select calls
            item_count = 0
            select_count = 0

            for event in events:
                name = event.name
                if "aten::item" in name:
                    item_count += 1
                elif "aten::select" in name:
                    select_count += 1

            # Check thresholds
            expected_item_calls = batch_size * self.item_threshold
            if item_count > expected_item_calls:
                ratio = item_count / expected_item_calls
                self.issues.append(
                    VectorizationIssue(
                        issue_type="scalar_iteration",
                        location=(module_name, 0),
                        function_name="forward",
                        description=(
                            f"Detected {item_count} aten::item calls, "
                            f"which is {ratio:.1f}x expected threshold"
                        ),
                        severity="performance",
                        suggestion="Scalar extraction (aten::item) breaks vectorization. Keep values as tensors.",
                        extra_info={
                            "item_count": item_count,
                            "expected_threshold": expected_item_calls,
                            "batch_size": batch_size,
                        },
                    )
                )

            # Check select operations
            expected_select_calls = batch_size * self.select_threshold
            if select_count > expected_select_calls:
                ratio = select_count / expected_select_calls
                self.issues.append(
                    VectorizationIssue(
                        issue_type="scalar_iteration",
                        location=(module_name, 0),
                        function_name="forward",
                        description=(
                            f"Detected {select_count} aten::select calls, "
                            f"which is {ratio:.1f}x expected threshold"
                        ),
                        severity="performance",
                        suggestion="Frequent tensor indexing may indicate non-vectorized code patterns.",
                        extra_info={
                            "select_count": select_count,
                            "expected_threshold": expected_select_calls,
                        },
                    )
                )

        except Exception as e:
            warnings.warn(f"Profiler analysis failed: {e}")

        return self.issues


# =============================================================================
# Recursion Depth Auditor
# =============================================================================


class RecursionDepthAuditor:
    """Monitor and audit recursive function calls."""

    def __init__(self, depth_time_ratio_threshold: float = 0.1):
        """
        Args:
            depth_time_ratio_threshold: If time increases linearly with depth,
                                        ratio will be ~1. If sublinear, ratio < 1.
        """
        self.depth_time_ratio_threshold = depth_time_ratio_threshold
        self.issues: List[VectorizationIssue] = []
        self._call_stack: List[Tuple[str, float]] = []
        self._call_times: Dict[str, List[float]] = {}

    @contextmanager
    def monitor_recursion(self, func_name: str):
        """Context manager to monitor a function call."""
        start_time = time.perf_counter()
        self._call_stack.append((func_name, start_time))

        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # Record timing
            if func_name not in self._call_times:
                self._call_times[func_name] = []
            self._call_times[func_name].append(elapsed)

            # Pop from stack
            self._call_stack.pop()

            # Check recursion depth
            depth = len(self._call_stack)
            if depth > 1:
                # Check if time scales linearly with depth
                if func_name in self._call_times and len(self._call_times[func_name]) > 1:
                    times = self._call_times[func_name]
                    if len(times) >= 2:
                        # Compare first call (depth=1) with current (depth>1)
                        base_time = times[0]
                        current_time = elapsed

                        if base_time > 0:
                            time_ratio = current_time / base_time
                            depth_ratio = depth / 1  # Compare to depth=1

                            # If time grows linearly with depth (ratio close to depth_ratio),
                            # this suggests non-vectorized recursion
                            if time_ratio > depth_ratio * self.depth_time_ratio_threshold * 2:
                                self.issues.append(
                                    VectorizationIssue(
                                        issue_type="recursion",
                                        location=("", 0),
                                        function_name=func_name,
                                        description=(
                                            f"Recursive call time scales linearly with depth. "
                                            f"Time ratio: {time_ratio:.2f}, Depth ratio: {depth_ratio:.2f}"
                                        ),
                                        severity="performance",
                                        suggestion="Recursion depth affects execution time linearly. "
                                                  "Consider iterative implementation.",
                                        extra_info={
                                            "depth": depth,
                                            "time_ratio": time_ratio,
                                            "depth_ratio": depth_ratio,
                                        },
                                    )
                                )

    def get_issues(self) -> List[VectorizationIssue]:
        """Get collected issues."""
        return self.issues

    def clear(self) -> None:
        """Clear collected issues and timing data."""
        self.issues = []
        self._call_times = {}


# =============================================================================
# Main Analyzer Class
# =============================================================================


class VectorizationAuditTool:
    """
    Main tool for comprehensive vectorization auditing.

    Combines static AST analysis, dynamic FX tracing, profiler analysis,
    and recursion depth auditing into a unified interface.
    """

    def __init__(
        self,
        enable_static: bool = True,
        enable_fx_tracing: bool = True,
        enable_profiler: bool = True,
        enable_recursion: bool = True,
        batch_size: int = 4,
    ):
        self.enable_static = enable_static
        self.enable_fx_tracing = enable_fx_tracing
        self.enable_profiler = enable_profiler
        self.enable_recursion = enable_recursion
        self.batch_size = batch_size

        self._static_analyzer = ASTVectorizationAnalyzer()
        self._fx_analyzer = FXTracingAnalyzer(batch_size=batch_size)
        self._profiler_detector = ProfilerScalarIteratorDetector()
        self._recursion_auditor = RecursionDepthAuditor()

    def analyze_module(
        self,
        module: Type,
        source_path: Optional[str] = None,
    ) -> VectorizationReport:
        """Perform comprehensive analysis on a module class."""
        start_time = time.perf_counter()
        report = VectorizationReport(module_name=module.__name__)

        if self.enable_static:
            issues = self._static_analyzer.analyze_module(module, source_path)
            for issue in issues:
                report.add_issue(issue)

        report.analysis_time = time.perf_counter() - start_time
        return report

    def analyze_model(
        self,
        model: torch.nn.Module,
        sample_input: Tensor,
        module_name: str = "unknown",
    ) -> VectorizationReport:
        """Perform dynamic analysis on a model instance."""
        start_time = time.perf_counter()
        report = VectorizationReport(module_name=module_name)

        if self.enable_fx_tracing:
            issues = self._fx_analyzer.analyze_forward(model, sample_input, module_name)
            for issue in issues:
                report.add_issue(issue)

        if self.enable_profiler:
            issues = self._profiler_detector.analyze_inference(
                model, sample_input, module_name, self.batch_size
            )
            for issue in issues:
                report.add_issue(issue)

        report.analysis_time = time.perf_counter() - start_time
        return report

    def analyze_component(
        self,
        component: Union[torch.nn.Module, Callable],
        *args,
        **kwargs,
    ) -> Tuple[VectorizationReport, Optional[Tensor]]:
        """
        Analyze a component (module or function) with provided inputs.

        Returns:
            Tuple of (report, output_tensor)
        """
        start_time = time.perf_counter()
        component_name = getattr(component, "__name__", component.__class__.__name__)
        report = VectorizationReport(module_name=component_name)

        # Try to trace and run
        try:
            if isinstance(component, torch.nn.Module):
                # Model mode
                input_tensor = args[0] if args else kwargs.get("input")
                if input_tensor is not None:
                    # FX tracing
                    if self.enable_fx_tracing:
                        issues = self._fx_analyzer.analyze_forward(
                            component, input_tensor, component_name
                        )
                        for issue in issues:
                            report.add_issue(issue)

                    # Profiler
                    if self.enable_profiler:
                        issues = self._profiler_detector.analyze_inference(
                            component, input_tensor, component_name, self.batch_size
                        )
                        for issue in issues:
                            report.add_issue(issue)

                    # Get output
                    with torch.no_grad():
                        output = component(input_tensor, **kwargs)
                else:
                    output = None

            else:
                # Function mode
                output = component(*args, **kwargs)

        except Exception as e:
            warnings.warn(f"Component analysis failed: {e}")
            output = None

        report.analysis_time = time.perf_counter() - start_time
        return report, output


# =============================================================================
# Decorator and Context Manager for Zero-Cost Auditing
# =============================================================================


def audit_vectorization(
    name: Optional[str] = None,
    enabled: Optional[bool] = None,
):
    """
    Decorator for auditing vectorization in a function or method.

    This decorator is zero-cost when auditing is disabled globally.

    Args:
        name: Optional name for the audit (defaults to function name)
        enabled: Override global audit setting

    Usage:
        @audit_vectorization("my_component")
        def forward(self, x):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Zero-cost: check enabled status early
            audit_enabled = enabled if enabled is not None else is_audit_enabled()
            if not audit_enabled:
                return func(*args, **kwargs)

            # Get audit tool
            auditor = VectorizationAuditTool()

            # For methods, get self to access module info
            self_arg = args[0] if args else None
            if hasattr(self_arg, "__class__"):
                pass

            # Run analysis
            report, _ = auditor.analyze_component(func, *args, **kwargs)

            # Report any issues
            if report.has_issues():
                for issue in report.get_performance_warnings():
                    warnings.warn(
                        f"[Vectorization Audit] {issue.function_name}:{issue.location[1]} - {issue.description}",
                        PerformanceWarning,
                        stacklevel=3,
                    )

            return func(*args, **kwargs)

        return wrapper
    return decorator


@contextmanager
def vectorization_audit(
    name: str,
    enabled: Optional[bool] = None,
):
    """
    Context manager for auditing vectorization in a code block.

    This context manager is zero-cost when auditing is disabled globally.

    Args:
        name: Name for the audit context
        enabled: Override global audit setting

    Usage:
        with vectorization_audit("tokenizer_forward"):
            output = model(input)
    """
    audit_enabled = enabled if enabled is not None else is_audit_enabled()

    if not audit_enabled:
        # Zero-cost: just pass through
        yield
        return

    # Import here to avoid circular imports
    from . import VectorizationAuditTool

    start_time = time.perf_counter()
    VectorizationAuditTool()
    issues_found: List[VectorizationIssue] = []

    try:
        yield
    finally:
        time.perf_counter() - start_time

    # Report findings
    if issues_found:
        for issue in issues_found:
            warnings.warn(
                f"[Vectorization Audit] {name}:{issue.location[1]} - {issue.description}",
                PerformanceWarning,
                stacklevel=2,
            )


class VectorizationAuditContext:
    """Context manager state holder."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0.0
        self.auditor = VectorizationAuditTool()

    def __enter__(self) -> "VectorizationAuditContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time

        # Report timing if significant
        if elapsed > 0.1:  # > 100ms
            warnings.warn(
                f"[Vectorization Audit] {self.name} took {elapsed*1000:.1f}ms - "
                "consider checking for vectorization opportunities.",
                PerformanceWarning,
                stacklevel=2,
            )

        return False  # Don't suppress exceptions


# =============================================================================
# Integration Helper for test_system.py
# =============================================================================


def run_vectorization_audit_on_model(
    model: torch.nn.Module,
    sample_input: Tensor,
    module_name: str = "model",
    batch_size: Optional[int] = None,
) -> VectorizationReport:
    """
    Run comprehensive vectorization audit on a model.

    Args:
        model: PyTorch model to audit
        sample_input: Example input tensor
        module_name: Name for the audit report
        batch_size: Batch size for threshold calculations

    Returns:
        VectorizationReport with all findings
    """
    batch_size = batch_size or sample_input.shape[0]
    report = VectorizationReport(module_name=module_name)

    # Run all analyzers
    static_analyzer = ASTVectorizationAnalyzer()
    fx_analyzer = FXTracingAnalyzer(batch_size=batch_size)
    profiler_detector = ProfilerScalarIteratorDetector()

    # Get source path if possible
    source_path = inspect.getfile(model.__class__) if hasattr(model, "__class__") else None

    # Static analysis
    if source_path:
        issues = static_analyzer.analyze_module(model.__class__, source_path)
        for issue in issues:
            report.add_issue(issue)

    # Dynamic analysis
    issues = fx_analyzer.analyze_forward(model, sample_input, module_name)
    for issue in issues:
        report.add_issue(issue)

    issues = profiler_detector.analyze_inference(model, sample_input, module_name, batch_size)
    for issue in issues:
        report.add_issue(issue)

    return report


def audit_and_report(
    model: torch.nn.Module,
    sample_input: Tensor,
    test_name: str = "test",
) -> List[PerformanceWarning]:
    """
    Run vectorization audit and emit warnings for any issues found.

    Args:
        model: Model to audit
        sample_input: Sample input tensor
        test_name: Name of the test for context

    Returns:
        List of emitted PerformanceWarning objects
    """
    warnings_list: List[PerformanceWarning] = []

    # Only run if audit is enabled
    if not is_audit_enabled():
        return warnings_list

    batch_size = sample_input.shape[0]
    report = run_vectorization_audit_on_model(
        model, sample_input, module_name=test_name, batch_size=batch_size
    )

    # Emit warnings for performance issues
    for issue in report.get_performance_warnings():
        warning = PerformanceWarning(
            f"[Vectorization Audit - {test_name}] {issue.function_name} at line {issue.location[1]}: "
            f"{issue.description}. Suggestion: {issue.suggestion}"
        )
        warnings_list.append(warning)
        warnings.warn(warning, stacklevel=2)

    # Emit errors for critical issues
    for issue in report.get_errors():
        warning = PerformanceWarning(
            f"[Vectorization Audit ERROR - {test_name}] {issue.description}"
        )
        warnings_list.append(warning)
        warnings.warn(warning, stacklevel=2)

    return warnings_list


# =============================================================================
# Convenience Functions for Test Integration
# =============================================================================


def quick_check_for_scalar_iteration(model: torch.nn.Module, input_tensor: Tensor) -> bool:
    """
    Quick check if a model has scalar iteration patterns.

    Returns:
        True if scalar iteration patterns detected
    """
    detector = ProfilerScalarIteratorDetector()
    issues = detector.analyze_inference(model, input_tensor, model.__class__.__name__)
    return len(issues) > 0


def check_module_for_loops(module_class: Type) -> List[Tuple[str, int]]:
    """
    Quick check for loops in a module class.

    Returns:
        List of (function_name, line_number) tuples for loops found
    """
    analyzer = ASTVectorizationAnalyzer()
    issues = analyzer.analyze_module(module_class)
    return [(i.function_name, i.location[1]) for i in issues if i.issue_type == "loop"]


# =============================================================================
# __init__.py exports
# =============================================================================

__all__ = [
    # Main tool
    "VectorizationAuditTool",
    "VectorizationReport",
    "VectorizationIssue",
    # Analyzers
    "ASTVectorizationAnalyzer",
    "FXTracingAnalyzer",
    "ProfilerScalarIteratorDetector",
    "RecursionDepthAuditor",
    # Integration helpers
    "run_vectorization_audit_on_model",
    "audit_and_report",
    "quick_check_for_scalar_iteration",
    "check_module_for_loops",
    # Decorators and context managers
    "audit_vectorization",
    "vectorization_audit",
    "VectorizationAuditContext",
    # Configuration
    "enable_vectorization_audit",
    "is_audit_enabled",
]
