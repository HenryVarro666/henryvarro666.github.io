"""Run one model demo with Python 3.12 and an OpenAI API key."""

import argparse
import json
import os
from typing import Literal

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

TITLE_INSTRUCTION = "把客服留言改写成一句简短的工单标题。"
TITLE_INPUT = "上周把货寄回去了，到现在还没看到状态更新。"


class Ticket(BaseModel):
    category: Literal["return", "refund_status", "other"]
    summary: str


class CitedAnswer(BaseModel):
    answer: str
    citations: list[str]


@tool
def lookup_policy(topic: Literal["return_window", "damaged_item"]) -> str:
    """Read one policy from the demo shop's support handbook."""
    policies = {
        "return_window": "[R1] 未使用的商品可在送达后 30 天内退货。",
        "damaged_item": "[R2] 商品有破损时请在 7 天内联系客服并附照片。",
    }
    return policies[topic]


def call_demo(model: ChatOpenAI) -> None:
    reply = model.invoke([
        SystemMessage(TITLE_INSTRUCTION),
        HumanMessage(TITLE_INPUT),
    ])
    print(reply.text)


def chain_demo(model: ChatOpenAI) -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", TITLE_INSTRUCTION),
        ("user", "{message}"),
    ])
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"message": TITLE_INPUT}))


def classify_demo(model: ChatOpenAI) -> None:
    extractor = model.with_structured_output(Ticket, method="json_schema")
    ticket = extractor.invoke("为这条留言提取类别和摘要：款什么时候退回银行卡？")
    print(ticket.model_dump())


def agent_demo(model: ChatOpenAI) -> None:
    agent = create_agent(
        model=model,
        tools=[lookup_policy],
        system_prompt="用中文回答。回答政策问题前先读取对应条款，保留条款编号。",
    )
    result = agent.invoke(
        {"messages": [HumanMessage("未使用的商品最晚什么时候可以退？")]},
        config={"recursion_limit": 12},
    )

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            print("Tool request:", message.tool_calls)
        elif isinstance(message, ToolMessage):
            print("Tool result:", message.tool_call_id, message.content)
    print(result["messages"][-1].text)


def rag_demo(model: ChatOpenAI) -> None:
    from offline_rag import DOCS, graph as knowledge_base

    question = "换手机了，怎么恢复 2FA？"
    search = knowledge_base.invoke({"query": question, "scope": "employee"})
    if search["abstained"]:
        print(search["answer"])
        return

    passages = {source_id: DOCS[source_id].page_content for source_id in search["retrieved_ids"]}
    writer = model.with_structured_output(CitedAnswer, method="json_schema")
    reply = writer.invoke([
        SystemMessage("用中文回答。仅使用 passages 里的证据，并填写所用片段的编号。"),
        HumanMessage(json.dumps({"question": question, "passages": passages}, ensure_ascii=False)),
    ])
    if reply is None or not reply.answer.strip() or not reply.citations:
        print("回答或引用缺失，转人工复核。")
    elif not set(reply.citations).issubset(passages):
        print("答案引用了本次未提供的资料，转人工复核。")
    else:
        print(reply.model_dump())


def main() -> None:
    demos = {
        "call": call_demo,
        "chain": chain_demo,
        "classify": classify_demo,
        "agent": agent_demo,
        "rag": rag_demo,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=demos)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your environment to run a model demo.")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
        timeout=30,
        max_retries=1,
    )
    demos[args.demo](model)


if __name__ == "__main__":
    main()
