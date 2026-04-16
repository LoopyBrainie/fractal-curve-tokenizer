"""血缘追踪工具 - 追踪指标派生关系"""


class LineageTracker:
    """追踪指标的血缘关系

    维护指标之间的父子派生关系，支持血缘追溯和影响分析。

    使用示例:
        tracker = LineageTracker()

        # 添加派生关系: parent -> child
        tracker.add_edge("loss", "total_loss")
        tracker.add_edge("grad_norm", "backbone_vs_splitter_ratio")

        # 获取指标的父指标
        parents = tracker.get_parents("total_loss")  # ["loss"]

        # 获取指标的子指标
        children = tracker.get_children("loss")  # ["total_loss"]

        # 追溯到根源叶子指标
        path = tracker.trace_back("backbone_vs_splitter_ratio")
        # 可能结果: ["backbone_vs_splitter_ratio", "grad_norm", "backbone_grad", "splitter_grad"]
    """

    def __init__(self):
        # 存储父子关系: parent -> [children]
        self._parents: dict[str, list[str]] = {}
        # 存储反向关系: child -> parent
        self._children: dict[str, list[str]] = {}

    def add_edge(self, parent: str, child: str) -> None:
        """添加血缘边: parent -> child

        Args:
            parent: 父指标名称
            child: 子指标名称
        """
        # 添加 parent -> child 关系
        if parent not in self._parents:
            self._parents[parent] = []
        if child not in self._parents[parent]:
            self._parents[parent].append(child)

        # 添加 child -> parent 反向关系
        if child not in self._children:
            self._children[child] = []
        if parent not in self._children[child]:
            self._children[child].append(parent)

    def get_parents(self, metric: str) -> list[str]:
        """获取指标的父指标列表

        Args:
            metric: 指标名称

        Returns:
            父指标名称列表
        """
        return self._children.get(metric, [])

    def get_children(self, metric: str) -> list[str]:
        """获取指标的子指标列表

        Args:
            metric: 指标名称

        Returns:
            子指标名称列表
        """
        return self._parents.get(metric, [])

    def trace_back(self, metric: str) -> list[str]:
        """追溯到根源叶子指标

        通过血缘链一直向上追溯，直到找到没有父指标的根源指标。

        Args:
            metric: 起始指标名称

        Returns:
            从起始指标到根源指标的路径列表
        """
        path = [metric]
        visited = {metric}

        current = metric
        while True:
            parents = self.get_parents(current)
            if not parents:
                break
            root = parents[0]  # 取第一个父指标
            if root in visited:
                # 检测到循环，停止追溯
                break
            path.append(root)
            visited.add(root)
            current = root

        return path

    def has_lineage(self, metric: str) -> bool:
        """检查指标是否有血缘（是否为派生指标）

        Args:
            metric: 指标名称

        Returns:
            True 如果有父指标（即为派生指标）
        """
        return metric in self._children and len(self._children[metric]) > 0

    def clear(self) -> None:
        """清空所有血缘关系"""
        self._parents.clear()
        self._children.clear()
