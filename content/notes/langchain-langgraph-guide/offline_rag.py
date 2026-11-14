from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

DOCS = {
    "KB-17#1": Document(
        page_content="在账户的 Security 页面重置双重验证。",
        metadata={"query": "重置双重验证", "scope": "employee"},
    ),
    "KB-42#1": Document(
        page_content="服务台管理员可在迁移面板中重置单点登录。",
        metadata={"query": "重置单点登录", "scope": "admin"},
    ),
}
ALIASES = {"换手机了，怎么恢复 2FA？": "重置双重验证"}
MAX_ATTEMPTS = 2


class State(TypedDict, total=False):
    query: str
    scope: str
    attempts: int
    retrieved_ids: list[str]
    answer: str
    citations: list[str]
    abstained: bool


def retrieve(state: State):
    ids = [
        source_id
        for source_id, doc in DOCS.items()
        if doc.metadata["scope"] == state["scope"]
        and doc.metadata["query"] == state["query"]
    ]
    return {"retrieved_ids": ids, "attempts": state.get("attempts", 0) + 1}


def next_step(state: State):
    if state["retrieved_ids"]:
        return "answer"
    if state["attempts"] < MAX_ATTEMPTS and state["query"] in ALIASES:
        return "rewrite"
    return "abstain"


def rewrite(state: State):
    return {"query": ALIASES[state["query"]]}


def answer(state: State):
    ids = state["retrieved_ids"]
    return {
        "answer": "\n".join(DOCS[source_id].page_content for source_id in ids),
        "citations": ids,
        "abstained": False,
    }


def abstain(state: State):
    return {"answer": "现有可见资料无法回答。", "citations": [], "abstained": True}


builder = StateGraph(State)
for node in (retrieve, rewrite, answer, abstain):
    builder.add_node(node)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", next_step, ["answer", "rewrite", "abstain"])
builder.add_edge("rewrite", "retrieve")
builder.add_edge("answer", END)
builder.add_edge("abstain", END)
graph = builder.compile()

if __name__ == "__main__":
    cases = [
        ("重置双重验证", 1, ["KB-17#1"], False),
        ("换手机了，怎么恢复 2FA？", 2, ["KB-17#1"], False),
        ("重置单点登录", 1, [], True),
        ("能代我修改账户吗？", 1, [], True),
    ]
    for query, attempts, citations, abstained in cases:
        result = graph.invoke({"query": query, "scope": "employee"})
        assert (result["attempts"], result["citations"], result["abstained"]) == (
            attempts, citations, abstained
        )
        print(result)
