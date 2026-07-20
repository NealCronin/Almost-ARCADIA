from __future__ import annotations

import json

from core.inference.results import LLMResult
from core.pipeline.priority_map_adapter import _GraphAgent


class Graph:
    def __init__(self):
        self.applied = []
        self.reviewed = []

    def count_unreviewed_nodes(self):
        return 2

    def get_agent_graph_data(self):
        return [("a", 10.0, 0), ("b", 20.0, 0)], [("a", "b", 3.0)], "spatial"

    def apply_score_delta(self, node_id, delta, *, view):
        self.applied.append((node_id, delta, view))
        return (10.0, 10.0 + delta)

    def mark_agent_reviewed(self, node_ids, *, view):
        self.reviewed.append((list(node_ids), view))


class Client:
    def chat(self, prompt, **kwargs):
        return LLMResult(json.dumps({"reasoning": "ok", "updates": [{"node_id": "a", "delta": 7}]}), {})


def test_graph_agent_synchronous_contract_and_mutation():
    graph = Graph()
    agent = _GraphAgent(graph, "Find vehicles", node_growth_threshold=1, llm_client=Client())
    assert agent.should_run()
    agent.update_priorities()
    assert graph.applied == [("a", 7.0, "spatial")]
    assert graph.reviewed == [(["a", "b"], "spatial")]
    agent.close()


def test_graph_agent_async_lifecycle():
    graph = Graph()
    agent = _GraphAgent(graph, "Find vehicles", node_growth_threshold=1, llm_client=Client())
    assert agent.start_async_if_ready()
    agent.close()
    assert graph.reviewed
