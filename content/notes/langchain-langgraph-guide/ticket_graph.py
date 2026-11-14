from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class Ticket(TypedDict):
    order_id: str
    request: str
    route: NotRequired[Literal["refund", "shipping", "manual"]]
    refund_cents: NotRequired[int]
    status: NotRequired[str]
    audit: Annotated[list[str], add]


REFUND_CENTS = {"A-104": 2400}  # Toy data: integer cents.


def triage(state: Ticket):
    if "退款" in state["request"]:
        route = "refund" if state["order_id"] in REFUND_CENTS else "manual"
    elif "物流" in state["request"]:
        route = "shipping"
    else:
        route = "manual"
    return {"route": route, "audit": [f"triage:{route}"]}


def draft_refund(state: Ticket):
    return {"refund_cents": REFUND_CENTS[state["order_id"]], "audit": ["draft"]}


def review(state: Ticket):
    decision = interrupt({
        "order_id": state["order_id"],
        "amount_cents": state["refund_cents"],
        "question": "是否批准此退款草稿？",
    })
    status = "draft_approved" if decision is True else "draft_rejected"
    return {"status": status, "audit": [status]}


def shipping(state: Ticket):
    return {"status": "show_tracking_help", "audit": ["shipping_help"]}


def manual(state: Ticket):
    return {"status": "needs_manual_review", "audit": ["manual_review"]}


graph = (
    StateGraph(Ticket)
    .add_node("triage", triage)
    .add_node("draft_refund", draft_refund)
    .add_node("review", review)
    .add_node("shipping", shipping)
    .add_node("manual", manual)
    .add_edge(START, "triage")
    .add_conditional_edges("triage", lambda state: state["route"], {
        "refund": "draft_refund", "shipping": "shipping", "manual": "manual",
    })
    .add_edge("draft_refund", "review")
    .add_edge("review", END)
    .add_edge("shipping", END)
    .add_edge("manual", END)
    .compile(checkpointer=InMemorySaver())
)

if __name__ == "__main__":
    ticket = {"order_id": "A-104", "request": "我想申请退款", "audit": []}
    for decision, expected in [(True, "draft_approved"), (False, "draft_rejected")]:
        config = {"configurable": {"thread_id": expected}}
        paused = graph.invoke(ticket, config)
        assert graph.get_state(config).next == ("review",)
        assert paused["__interrupt__"][0].value["amount_cents"] == 2400
        print(paused["__interrupt__"][0].value)
        final = graph.invoke(Command(resume=decision), config)
        assert final["status"] == expected
        assert final["audit"] == ["triage:refund", "draft", expected]
        assert graph.get_state(config).next == ()
        print(final["status"], final["audit"])

    final = graph.invoke(
        {**ticket, "request": "想查物流"},
        {"configurable": {"thread_id": "shipping-demo"}},
    )
    assert final["status"] == "show_tracking_help"
    assert "__interrupt__" not in final
    final = graph.invoke(
        {**ticket, "order_id": "unknown"},
        {"configurable": {"thread_id": "manual-demo"}},
    )
    assert final["status"] == "needs_manual_review"
    assert "__interrupt__" not in final
