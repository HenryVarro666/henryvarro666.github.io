---
title: "用客服工单学 LangChain 与 LangGraph"
date: 2026-11-13T19:40:00-06:00
description: "用 Python 跑通模型和只读工具，搭一个能暂停审批的工单流程，再动手做带引用和拒答的知识库。"
topics: [llm, ml-systems]
---

“未使用的商品还能退吗？”查手册就能回答。“帮我退订单 A-104 的钱”则要查订单、起草退款单、等主管批准。两条消息可以进同一个客服收件箱，处理方式却不同。本文用一家虚构商店的客服台和内部帮助中心做练习。从调模型、查政策，到让退款草稿停在审批节点，最后做一个能给出处、查不到资料就拒答的知识库。

读懂 Python 函数和字典、用过命令行，就能跟做。第一次接触大语言模型（large language model，LLM）的应用开发，可以从[搭环境](#setup)和[第一次调用](#langchain)开始。想先把流程跑通，可以直接看[离线工单](#graph)和[知识库实验](#rag)，它们只使用模拟数据。

## 什么时候用 LangChain，什么时候画图 {#choose}

**LangChain** 提供模型、消息、提示模板和工具的 Python 接口。用 `create_agent` 创建的智能体（agent）在 LangGraph 上运行。**LangGraph** 负责按图执行工作流，图里写清楚哪些状态要保留、下一步由谁决定。一个节点可以用 `create_agent`，也可以只调用一次模型或执行普通 Python 函数。 [LangChain 概览](https://docs.langchain.com/oss/python/langchain/overview) · [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)

| 想做什么 | 从哪里开始 | 谁决定下一步 |
|---|---|---|
| 把客服留言压成工单标题 | 服务商的软件开发工具包（SDK），或 LangChain 聊天模型 | 应用发起一次模型调用 |
| 每条留言都生成标题，再套上固定格式 | 普通 Python 函数，或 LangChain 的 Runnable 序列 | 应用代码规定步骤顺序 |
| 根据问题决定查手册还是查订单 | `create_agent` 加一组授权工具 | 模型提出工具请求，应用限制可用工具和调用次数 |
| 退款、物流、转人工各走不同流程 | `StateGraph` | 图按业务规则或明确的决策走对应的边 |

固定流程中也可以用模型。比如让模型提取 `refund_status` 意图，再由业务代码验证订单归属、选择下一步。只有查询路径随问题变化、适合让模型决定去哪里取信息时，才需要让模型进入工具循环。 [工作流与智能体模式](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

## 搭好 Python 环境 {#setup}

下载 [requirements.txt](/notes/langchain-langgraph-guide/requirements.txt)、[langchain_basics.py](/notes/langchain-langgraph-guide/langchain_basics.py)、[ticket_graph.py](/notes/langchain-langgraph-guide/ticket_graph.py) 和 [offline_rag.py](/notes/langchain-langgraph-guide/offline_rag.py)，保存到同一目录。在终端进入该目录。示例使用 Python 3.12，固定 LangChain 1.5.0、LangChain Core 1.8.0、`langchain-openai` 1.8.0 和 LangGraph 1.2.21。更换依赖时，先看对应发行版的 Python 版本要求。 [LangChain 的 PyPI 页面](https://pypi.org/project/langchain/1.5.0/) · [LangGraph 的 PyPI 页面](https://pypi.org/project/langgraph/1.2.21/) · [OpenAI 集成包的 PyPI 页面](https://pypi.org/project/langchain-openai/1.8.0/)

macOS 或 Linux 可以在终端运行下面几行。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
```

Windows 可以用 PowerShell 运行对应命令。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
$env:LANGSMITH_TRACING = "false"
$env:LANGCHAIN_TRACING_V2 = "false"
```

装好依赖后，`python ticket_graph.py` 和 `python offline_rag.py` 可以直接运行，无需密钥。可选的 `langchain_basics.py` 演示会调用 OpenAI 的应用程序接口（API），可能产生费用。运行前在本机环境设置 `OPENAI_API_KEY`，密钥留在环境或密钥管理器里。脚本默认选 `gpt-5.4-mini`。如果需要换模型，将 `OPENAI_MODEL` 设为账号能访问且支持工具调用和结构化输出的模型。 [OpenAI 模型能力](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [ChatOpenAI 集成](https://docs.langchain.com/oss/python/integrations/chat/openai)

在 macOS 或 Linux 终端运行下面一行，按隐藏提示输入密钥，再在同一终端运行模型示例。

```bash
export OPENAI_API_KEY="$(python -c 'from getpass import getpass; print(getpass("OpenAI API key: "))')"
```

PowerShell 可以这样隐藏输入的密钥。

```powershell
$key = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
```

碰到旧教程的 `langgraph.prebuilt.create_react_agent`，可以对照 1.x 迁移文档里的 `langchain.agents.create_agent`。旧教程中的 `LLMChain` 归在 `langchain-classic` 包。另一个常见混淆是包名与导入名，安装用 `langchain-openai`，导入用 `langchain_openai`。 [LangChain v1 迁移文档](https://docs.langchain.com/oss/python/migrate/langchain-v1)

## 调一次模型，再给它一个工具 {#langchain}

### 读懂第一次回复 {#model-call}

运行 `python langchain_basics.py call`，或在设好密钥的 Python 会话里运行下面这段。系统消息规定任务，用户消息提供素材。模型返回一条 `AIMessage`，里面的 `.text` 属性可以直接取文本。绑定工具以后，也可以用 `.tool_calls` 查看模型提出的工具请求。普通的模型调用接收此次传入的消息。想接着聊，就由应用把旧消息和新问题一起传入。 [聊天模型](https://docs.langchain.com/oss/python/langchain/models) · [消息类型](https://docs.langchain.com/oss/python/langchain/messages)

```python
import os

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), timeout=30)
reply = model.invoke([
    SystemMessage("把客服留言改写成一句简短的工单标题。"),
    HumanMessage("上周把货寄回去了，到现在还没看到状态更新。"),
])
print(reply.text)
```

可能得到“已寄回商品，退货状态未更新”。如果很多留言都要做同样的处理，就把提示做成 `ChatPromptTemplate`，后面接模型和解析器。这个写法叫 LangChain Expression Language（LCEL）。`|` 把一步的输出交给下一步。`invoke` 处理一个输入，`batch` 接收多个输入。下载脚本里可以用 `python langchain_basics.py chain` 看完整例子。 [RunnableSequence 参考](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)

```python
# 沿用上一个例子中的 model。
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "把客服留言改写成一句简短的工单标题。"),
    ("user", "{message}"),
])
chain = prompt | model | StrOutputParser()
print(chain.invoke({"message": "上周把货寄回去了，到现在还没看到状态更新。"}))
```

### 让下游程序直接读字段 {#structured-output}

工单标题给人看就够了。下一步要按 `category` 分流时，用 schema 把字段和取值写清楚。下面的 `Literal` 把类别限制在三个值里，Pydantic 校验解析后的对象。可以用 `python langchain_basics.py classify` 运行。示例里问银行卡到账时间，预期类别是 `refund_status`。类型校验只管结构，要判断分类是否正确，还得给几条容易混淆的留言做人工标注。在应用边界处理拒绝响应、解析错误和超时。需要排查格式时，可以设置 `include_raw=True`，拿到 `raw`、`parsed` 和 `parsing_error`。 [模型的结构化输出](https://docs.langchain.com/oss/python/langchain/models#structured-output) · [`ChatOpenAI.with_structured_output` 参考](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output)

```python
# 沿用上一个例子中的 model。
from typing import Literal

from pydantic import BaseModel

class Ticket(BaseModel):
    category: Literal["return", "refund_status", "other"]
    summary: str

extractor = model.with_structured_output(Ticket, method="json_schema")
ticket = extractor.invoke("为这条留言提取类别和摘要：款什么时候退回银行卡？")
print(ticket.model_dump())
```

结构化输出给调用方一组可读字段。工具调用则让运行时带着参数执行一个操作。智能体可以同时使用这两件事。比如指定 `create_agent(response_format=Ticket, ...)`，最后从 `structured_response` 读解析结果。某些模型会借用函数调用协议返回这些字段，业务工具仍通过应用的工具层执行。 [智能体的结构化输出](https://docs.langchain.com/oss/python/langchain/structured-output)

### 跟着消息看清楚谁在查手册 {#tool-call}

下面的 `lookup_policy` 有类型标注和 docstring，`@tool` 把它们变成模型可见的工具参数与说明。先自己用 `.invoke()` 查一条，确认条款和编号。接着复用前面的 `model`，让智能体用同一个工具。下载脚本的 `python langchain_basics.py agent` 会打印工具请求、返回结果和最终回答。 [定义工具](https://docs.langchain.com/oss/python/langchain/tools) · [智能体](https://docs.langchain.com/oss/python/langchain/agents)

```python
from typing import Literal

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

@tool
def lookup_policy(topic: Literal["return_window", "damaged_item"]) -> str:
    """Read one policy from the demo shop's support handbook."""
    policies = {
        "return_window": "[R1] 未使用的商品可在送达后 30 天内退货。",
        "damaged_item": "[R2] 商品有破损时请在 7 天内联系客服并附照片。",
    }
    return policies[topic]

print(lookup_policy.invoke({"topic": "return_window"}))
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
    print(message.type, message)
```

一次典型的消息顺序是 `human → ai → tool → ai`。第一条 `AIMessage` 可能没有文本，里面只有 `tool_calls`。假设其中的请求 ID 是 `call_demo`，运行时带 `return_window` 参数执行工具，把得到的条款装进 `ToolMessage`，并填入相同的 `tool_call_id`。下一次模型调用同时拿到工具请求和执行结果。模型可能一次请求几个工具，也可能继续调用或直接回答。要判断某次回答是否查过手册，先看消息记录。如果业务要求每次答政策都查，把检索设成工作流中的必经步骤。示例中的 `recursion_limit` 限制图的执行步数。服务还可以通过中间件单独限制模型或工具的调用次数。 [模型的工具调用](https://docs.langchain.com/oss/python/langchain/models#tool-calling) · [工具消息](https://docs.langchain.com/oss/python/langchain/messages#tool-message) · [内置中间件](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

![模型用 call_demo 请求 lookup_policy，Python 执行函数，并用同一个 ID 把 R1 条款作为 ToolMessage 交回去。模型根据该条款回答。](tool-loop.svg "图 1｜模型提出查询请求，应用执行函数。请求和工具结果用同一个 ID 对上。")

## 用离线图处理一张工单 {#graph}

运行 `python ticket_graph.py`。这个脚本用普通 Python 函数走四条路径。`triage` 读取留言，写入 `route`。退款分支从模拟订单 `A-104` 取金额，生成一张 2,400 美分的退款草稿，然后停在审核处。物流留言会收到查询指引，查不到的订单退款则转人工。脚本只记录草稿决定和处理状态。

![工单分流后，已知订单退款进入起草和审批，物流问题进入查询指引，未知订单或其他请求转人工。](ticket-routes.svg "图 2｜一条条件边选择后续路径。退款分支先起草，再等审核。")

- **状态（state）**是跨步骤共享的字段。这里的 `Ticket` 声明订单号、留言、路由、退款金额和 `audit` 等字段。
- **节点（node）**读取状态，返回需要更新的字段。比如 `draft_refund` 读取模拟订单，返回金额和一条审计记录。
- **边（edge）**决定下一步。`START → triage` 是固定边。`triage` 的条件边根据刚写好的 `route` 选择路径。走到 `END` 就结束。`compile()` 把声明编译成可以 `invoke`、`stream` 的对象。 [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

下面是连线方式，`Ticket`、节点函数和导入见完整文件。

```python
builder = StateGraph(Ticket)
for node in (triage, draft_refund, review, shipping, manual):
    builder.add_node(node)
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", lambda state: state["route"], {
    "refund": "draft_refund",
    "shipping": "shipping",
    "manual": "manual",
})
builder.add_edge("draft_refund", "review")
for terminal in ("review", "shipping", "manual"):
    builder.add_edge(terminal, END)
graph = builder.compile(checkpointer=InMemorySaver())
```

普通字段每次更新时覆盖旧值。`audit` 用 `audit: Annotated[list[str], add]` 指定合并函数（reducer），其中 `add` 来自 `operator`。节点只返回新增记录，例如 `{"audit": ["draft"]}`，运行时把新旧列表拼起来。返回完整的旧列表加新条目，会把历史再累计一遍。并行节点如果在同一执行轮（super-step）写同一个普通字段，就要给该字段定义 reducer，或改写不同字段再汇总。顺序有业务含义时，给条目加序号，汇总时显式排序。会话状态可以使用 `MessagesState`，它通过 `add_messages` 合并消息，并允许按消息 ID 更新。 [状态与 reducer](https://docs.langchain.com/oss/python/langgraph/graph-api-state) · [并发写同一字段的错误](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE)

| `ticket_graph.py` 的路径 | 脚本断言 |
|---|---|
| `A-104` 退款，批准 | 金额 2,400 美分，在 `review` 暂停。恢复后状态为 `draft_approved`，共有三条审计记录 |
| `A-104` 退款，驳回 | 同样在 `review` 暂停。恢复后状态为 `draft_rejected` |
| “想查物流” | 直接得到 `show_tracking_help` |
| 未知订单申请退款 | 直接得到 `needs_manual_review` |

练习时可以加一条“商品破损”支路，先断言终点对不对，再试着用模型的结构化输出代替关键词分类。订单是否存在、金额是否合法、能否退款，仍由应用判断。如果一张工单里的附件数量要运行时才知道，可以让路由返回多个 `Send`，每个任务处理一份附件，再通过 reducer 收集结果。 [Send 与 Command](https://docs.langchain.com/oss/python/langgraph/graph-api-send-command)

## 保存进度，恢复同一张工单 {#persistence}

**检查点保存器（checkpointer）**在图的执行边界保存快照。`thread_id` 用来选中本次工单的执行历史。审批时传回同一个 ID，办理另一张工单时用新 ID，避免输入与累计字段混进旧状态。`InMemorySaver` 的数据只留在当前进程。单机练习要跨重启，可以试 SQLite 检查点。多实例服务可以评估 PostgreSQL 等数据库实现。**Store** 存的是跨 thread 读取的应用数据，例如员工的语言偏好。按账号或租户组织和限制访问。 [持久化](https://docs.langchain.com/oss/python/langgraph/persistence) · [检查点保存器](https://docs.langchain.com/oss/python/langgraph/checkpointers) · [Store](https://docs.langchain.com/oss/python/langgraph/stores)

`review` 中的 `interrupt(payload)` 把订单和金额交给调用方。用同一个 `thread_id` 发送 `Command(resume=True)`，`True` 就会成为节点里 `interrupt()` 的返回值。在练习目录打开 Python 会话，把下面两次调用留在同一个进程里。

```python
from langgraph.types import Command
from ticket_graph import graph

config = {"configurable": {"thread_id": "try-a-104"}}
paused = graph.invoke(
    {"order_id": "A-104", "request": "我想申请退款", "audit": []},
    config,
)
print(paused["__interrupt__"][0].value)  # 待审核的内容
print(graph.get_state(config).next)     # ('review',)
final = graph.invoke(Command(resume=True), config)
print(final["status"], final["audit"])
# draft_approved ['triage:refund', 'draft', 'draft_approved']
```

![工单先保存退款草稿，再停在 review。恢复时使用同一 thread，review 从第一行重跑，最终记录草稿批准或驳回。](checkpoint-resume.svg "图 3｜恢复时沿用原 thread。review 从节点开头重跑，跑到 interrupt() 时读到审批决定。")

**一接真实操作，就要考虑节点重跑。** `review` 恢复时从第一行开始执行。起草放上一个节点，真实退款要经过审批，再由独立节点执行。把审批绑定到具体草稿版本，在服务端核对审批者身份、角色和工单权限。假设支付系统已收下退款请求，应用却因超时没收到回执，重试时应沿用同一个操作键，例如已审批退款的唯一 ID。保存外部执行结果，状态不明时先向支付系统对账。检查点负责保存图里的进度，外部操作能否安全重试还取决于写入接口的幂等设计。 [中断执行规则](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts) · [容错](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

`get_state(config).values` 能看保存的字段，`.next` 能看待运行的节点，`get_state_history(config)` 能看历史快照。从较早的检查点重放可能再次执行后面的节点，用模拟数据或只读工具先查清路径。网络请求只对适合重试的错误设置有限重试。循环要把业务尝试次数写入状态，并为超限设计一个能到 `END` 的分支。 [历史重放](https://docs.langchain.com/oss/python/langgraph/use-time-travel) · [图执行步数上限](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)

### 逐步看节点返回了什么 {#stream}

接着在同一个 Python 会话里跑一条物流工单，给它分配新 thread。`version="v2"` 的 `updates` 事件在 `data` 里给出节点更新。要看完整状态可用 `values`，要显示模型片段可用 `messages`，要显示“正在查手册”这类业务进度可用 `custom`。前端只发送调用者可见的事件和字段。 [流式输出模式](https://docs.langchain.com/oss/python/langgraph/streaming)

```python
for part in graph.stream(
    {"order_id": "A-104", "request": "想查物流", "audit": []},
    config={"configurable": {"thread_id": "shipping-stream"}},
    stream_mode="updates",
    version="v2",
):
    if part["type"] == "updates":
        print(part["data"])
```

这里会先看到 `triage`，再看到 `shipping`。真实审批可以把“等待审核”和“继续处理”做成两次请求。保存好检查点，等待期间就可以关掉流式连接。

## 做一个带来源和拒答的内部知识库 {#rag}

商店的内部帮助中心有两份资料。员工可以看双重验证（two-factor authentication，2FA）的恢复指引，服务台管理员还可以看单点登录（single sign-on，SSO）的重置流程。员工问“换手机了，怎么恢复 2FA？”时，第一次搜不到，可以换个说法再找。员工问管理员专用操作时，可以告诉他“现有可见资料无法回答”。

**检索增强生成（retrieval-augmented generation，RAG）**可以先从两步开始，按问题检索，再把证据交给模型写答复。整理真实手册时，按章节和内容切出可以单独引用的片段，为每段保留稳定 ID、文档标题、原文链接、版本和权限范围。检索时先把范围限定在调用者能读到的资料，把命中的原文交给模型，再附上能核对的出处。向量检索适合按语义找相似内容。错误码、型号和政策编号也值得用关键词检索。 [LangChain 语义搜索与最小 RAG](https://docs.langchain.com/oss/python/langchain/knowledge-base)

运行 `python offline_rag.py`，看如何在第一次没结果时补查一次。脚本用两个内存中的 `Document`、确定性的字符串匹配和一条改写词典来模拟检索。答复直接摘取经过权限过滤的资料。接入向量服务和模型之前，可以先确认每条路径走对了。

![检索在权限范围内找片段。命中后返回答案和来源 ID。未命中且有改写额度时再检索一次，否则拒答。](rag-loop.svg "图 4｜首轮和补查都要过权限过滤。只有还有一次改写机会且存在改写词时，才会补查。")

| 员工身份下的输入 | 检索次数 | 预期结果 |
|---|---:|---|
| `重置双重验证` | 1 | 返回答案，引用 `KB-17#1` |
| `换手机了，怎么恢复 2FA？` | 2 | 改写后命中，引用 `KB-17#1` |
| `重置单点登录` | 1 | 拒答。资料仅管理员可见 |
| `能代我修改账户吗？` | 1 | 拒答。可见资料没有合适答案 |

`retrieve` 的条件边调用 `next_step`。有命中就回答，没命中再判断是否还可以改写。下面的 `MAX_ATTEMPTS = 2` 包含首次检索。

```python
def next_step(state: State):
    if state["retrieved_ids"]:
        return "answer"
    if state["attempts"] < MAX_ATTEMPTS and state["query"] in ALIASES:
        return "rewrite"
    return "abstain"
```

第二次仍按同一权限范围查询。到 `abstain` 就返回说明和空引用。四组断言逐一检查检索次数、引用和拒答标记。 [LangGraph 自定义 RAG 教程](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

有模型密钥的话，可以运行 `python langchain_basics.py rag`，在同一份检索练习后接上生成。脚本把原问题和命中的授权片段交给模型。预期答复含 `answer` 和 `citations: ["KB-17#1"]`。脚本检查答案和引用是否齐全、引用编号是否出自本次提供的片段。可以再读一遍原文，核对它是否支持模型给出的具体步骤。

### 换成真实文档，保留三处约束 {#rag-upgrade}

1. **保留固定两步版作基线。** 从一轮检索加一轮生成开始。把原问题和 `search_query` 分开存，改写后的词给检索用，回答仍围绕原问题。是否值得加改写调用，可以用同一批问题对照试验。
2. **在证据进模型之前校验权限。** 示例里的 `scope` 是模拟值。服务端从已认证的身份取得权限范围，在检索器内落实。知识库原文也可能夹带提示注入。把它当作证据，限制模型可用工具，并加一条“文档诱导模型进行越权操作”的测试。 [OWASP 的间接提示注入说明](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
3. **校验回答和出处。** 可以让模型返回 `answer` 与 `citations`。有事实性回答至少给一个引用，每个 ID 要在本次真正送进模型的授权片段里。前端链接用可信的来源元数据生成。ID 合法后，还要核对片段能否支持这句话。有效版本之间有冲突，或缺一条关键依据时，可以转人工。

## 把流程走通和回答质量分开测 {#evaluation}

先写工具参数和节点规则的函数测试。给一个假模型预设工具请求，断言系统确实执行了对应工具，且返回的 `tool_call_id` 对得上。对图，检查直接结束、暂停后批准、暂停后拒绝和越权请求。应用有外部写入时，模拟“服务端收到了请求，客户端超时”，验证同一操作重试时不会重复执行。 [LangChain 单元测试](https://docs.langchain.com/oss/python/langchain/test/unit-testing) · [LangGraph 测试文档](https://docs.langchain.com/oss/python/langgraph/test)

再做一份固定样本集。每条记录写 `question`、调用者权限、标注的来源 ID、参考答案或评分规则，以及是否应该拒答。加上同义改写、需要多个片段的问题、互相冲突的版本和跨权限请求。每次运行记下提示、模型和索引的版本。 [LangSmith 评估概念](https://docs.langchain.com/langsmith/evaluation-concepts) · [RAG 评估教程](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)

| 指标 | 计数口径 |
|---|---|
| Hit@k | 在有标注来源的可答问题中，前 `k` 条至少命中一个标注来源的问题数 ÷ 这批问题总数 |
| Recall@k | 对同一批问题，逐题算前 `k` 条覆盖了多少标注来源，再对这些覆盖比例求均值 |
| 引用 ID 合法率 | 本次确实交给模型且通过授权的引用个数 ÷ 返回的引用个数。有事实性回答却无引用的，另记缺失 |
| 证据支持率 | 能由所引片段支持的事实性断言数 ÷ 全部可核查的事实性断言数。由人工评分，或先用人工样本校准自动评分 |
| 拒答召回率 | 正确拒答的样本数 ÷ 应该拒答的样本总数 |
| 误拒率 | 被拒答的可答样本数 ÷ 可答样本总数 |

例如，一道题需要 A 和 B 两段证据，检索只找到 A。该题算一次命中，召回率是 `1/2`。有补查时，预先约定按哪个候选集计分，例如按最终给模型的前 `k` 段。记录分子、分母，分母为零就标不可用。对比“只查一次”和“允许补查一次”时，尽量保持其他部分一致，同时记录中位数和第 95 百分位响应时间、工具和模型调用次数、输入输出 token 数。这样能看出补查获得的收益和代价。

一次请求的 trace 可以看成各步骤的执行记录。从节点名、耗时、分支、调用次数和允许记录的来源 ID 开始留日志。使用 LangSmith 等服务接收真实流量之前，先检查哪些输入、输出和元数据需要脱敏。选出典型失败，把经过审核的样本补进离线回归集。 [LangChain 追踪](https://docs.langchain.com/langsmith/trace-with-langchain) · [追踪数据脱敏](https://docs.langchain.com/langsmith/mask-inputs-outputs)

| 遇到的问题 | 先看哪里 |
|---|---|
| `ModuleNotFoundError` 或照着教程导入失败 | 检查当前解释器、`python -m pip check` 和迁移文档。检查本地有没有脚本叫 `langchain.py` 或 `langgraph.py` |
| 工具报错或回答编了条款 | 查看工具请求 ID 和参数，直接调用一次工具，核对 `ToolMessage` 的内容 |
| 恢复跑错工单或找不到检查点 | 核对 `thread_id`、保存器是否仍可访问，以及快照里有没有待审核节点 |
| 图超过步数限制 | 打印分支和尝试次数，找出应该通向 `END` 的出口 |
| 明明有相关资料却没召回 | 按片段 ID 查权限和版本，再看改写词和候选排序 |
| 找到了资料，答案仍不对 | 查实际送给模型的段落，看引用能否支撑答复里的关键句 |

## 推荐学习顺序与外部资料 {#resources}

1. **先学消息、模型和工具。** 跟着 Python 课程跑消息、工具和结构化输出，把案例换成一个只接受单个参数的工具，再从消息里找对应的 `ToolMessage`。 {{< new-tab "LangChain Essentials（Python）" "https://academy.langchain.com/courses/langchain-essentials-python" >}}
2. **手写一个可恢复的图。** 学节点、条件边、状态和中断。跑完短课，再读 101 Notebook 和它的邮件分流源码，观察哪些判断写在图里。 {{< new-tab "LangGraph Essentials（Python）" "https://academy.langchain.com/courses/langgraph-essentials-python" >}} · {{< new-tab "LangGraph 101 Notebook" "https://github.com/langchain-ai/langgraph-101" >}} · {{< new-tab "邮件智能体源码" "https://github.com/langchain-ai/langgraph-101/blob/main/agents/email_agent/graph.py" >}}
3. **给一份小手册做带引用的查询。** 用公开手册建索引，挑五个问题点回原文核对。再加最多一次改写，用同一组问题比较来源命中和误拒。 {{< new-tab "语义搜索与最小 RAG" "https://docs.langchain.com/oss/python/langchain/knowledge-base" >}} · {{< new-tab "用 LangGraph 搭自定义 RAG" "https://docs.langchain.com/oss/python/langgraph/agentic-rag" >}} · {{< new-tab "用 LangSmith 评估 RAG" "https://docs.langchain.com/langsmith/evaluate-rag-tutorial" >}}
4. **有明确需求再扩展控制流。** 先读固定链、路由和并行的模式图。想做只读的论文调查助手，可以从一个智能体和明确的调用上限开始。遇到不同工具或权限边界，或多项任务确实要并行时，再尝试拆成多个 worker。 {{< new-tab "Building effective agents" "https://www.anthropic.com/research/building-effective-agents" >}} · {{< new-tab "ReAct 论文" "https://arxiv.org/abs/2210.03629" >}} · {{< new-tab "LangGraph 工作流模式" "https://docs.langchain.com/oss/python/langgraph/workflows-agents" >}}
