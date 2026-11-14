---
title: "Learn LangChain and LangGraph by building a help desk"
date: 2026-11-13T19:40:00-06:00
description: "Start with a Python model call and a read-only tool. Build a ticket workflow that pauses for approval, then test a knowledge base that cites its sources."
topics: [llm, ml-systems]
---

“Can I still return this?” is a policy question. “Refund order A-104” needs an order lookup, a draft, and a person's approval. Both can land in the same support inbox. The fictional shop in this note has a customer support desk and an internal help center for its staff. You will use them to work through a model call, a tool-calling agent, a resumable ticket workflow, and a knowledge base that can decline to answer.

You need basic Python functions, dictionaries, and command-line skills. Start with [setup](#setup) and [the first model call](#langchain), or jump straight to the [offline ticket workflow](#graph) and [knowledge-base lab](#rag) if you want to begin without a model account.

## Choosing how much framework you need {#choose}

**LangChain** gives you Python interfaces for language models, messages, prompts, and tools. Its `create_agent` builds a loop that runs on LangGraph. **LangGraph** provides the runtime for a workflow with explicit state and execution paths. A LangGraph node can call `create_agent`, call a model directly, or run an ordinary Python function. [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview) · [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)

| Task | Starting point | What chooses the next step |
|---|---|---|
| Turn a customer message into a title | Provider software development kit (SDK) or a LangChain chat model | The application makes one model call |
| Turn each message into a title, then format it | Python functions or a LangChain Runnable sequence | A fixed order in application code |
| Let a model decide whether to look up a policy or an order | `create_agent` with a small set of authorized tools | The model requests tools. The application decides which tools are available and enforces limits |
| Route refund, shipping, and manual review tickets through different steps | `StateGraph` | An edge follows a business rule or an explicit decision |

A model can extract an intent inside a workflow with fixed rules. The model might return `refund_status`, for example, while application code checks whose order it is and selects the next node. Add a model-driven tool loop when several paths are plausible and you want the model to choose which information to gather. [Workflow and agent patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

## Set up a Python environment {#setup}

Download [requirements.txt](/notes/langchain-langgraph-guide/requirements.txt), [langchain_basics.py](/notes/langchain-langgraph-guide/langchain_basics.py), [ticket_graph.py](/notes/langchain-langgraph-guide/ticket_graph.py), and [offline_rag.py](/notes/langchain-langgraph-guide/offline_rag.py) to the same folder. Open a terminal in that folder. The examples use Python 3.12 and pin LangChain 1.5.0, LangChain Core 1.8.0, `langchain-openai` 1.8.0, and LangGraph 1.2.21. Check the Python requirement for each package release if you change versions. [LangChain on PyPI](https://pypi.org/project/langchain/1.5.0/) · [LangGraph on PyPI](https://pypi.org/project/langgraph/1.2.21/) · [Provider package on PyPI](https://pypi.org/project/langchain-openai/1.8.0/)

Use the following commands in a macOS or Linux terminal.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
```

On Windows, run the equivalent commands in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
$env:LANGSMITH_TRACING = "false"
$env:LANGCHAIN_TRACING_V2 = "false"
```

After installing the packages, `python ticket_graph.py` and `python offline_rag.py` run locally with synthetic data and need no key. The optional `langchain_basics.py` commands use the OpenAI application programming interface (API) and may incur charges. Set `OPENAI_API_KEY` in your local environment before running those commands. Use the default `gpt-5.4-mini` or set `OPENAI_MODEL` to a model you can access that supports tool calling and structured outputs. [OpenAI model capabilities](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [ChatOpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai)

In a macOS or Linux terminal, the following command puts the key in the current shell from a hidden prompt. Run the model examples from that session.

```bash
export OPENAI_API_KEY="$(python -c 'from getpass import getpass; print(getpass("OpenAI API key: "))')"
```

In PowerShell, use this command for the hidden prompt.

```powershell
$key = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
```

If an old tutorial uses `langgraph.prebuilt.create_react_agent`, look for `langchain.agents.create_agent` in the 1.x migration guide. Older `LLMChain` examples belong to the classic package. For the examples here, `langchain-openai` is the name you install and `langchain_openai` is the name you import. [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1)

## Call a model, then let it use one tool {#langchain}

### Read the first response {#model-call}

Run `python langchain_basics.py call` or paste the following example into a Python session. The downloadable script uses Chinese messages for the same support cases. The system message defines the job and the human message supplies text. The model returns an `AIMessage`. Its `.text` property gives you the text to show a reader, while `.tool_calls` exposes any requests to use tools in a tool-enabled call. A direct model call gets the messages you pass it. To continue a conversation, pass the prior messages along with the new question. [Chat models](https://docs.langchain.com/oss/python/langchain/models) · [Message roles](https://docs.langchain.com/oss/python/langchain/messages)

```python
import os

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), timeout=30)
reply = model.invoke([
    SystemMessage("Summarize the support message as a short ticket title."),
    HumanMessage("I mailed the return last week and the status still hasn't changed."),
])
print(reply.text)
```

`Return status still pending` is one possible result. If you need to process several of these, turn the instructions into a `ChatPromptTemplate` and put a model and a parser after it. This is LangChain Expression Language (LCEL). The `|` operator passes one component's output into the next. `invoke` takes one input and `batch` takes several. Use `python langchain_basics.py chain` to see the sequence in the downloadable script. [RunnableSequence reference](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)

```python
# Continue with the model from the first example.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the support message as a short ticket title."),
    ("user", "{message}"),
])
chain = prompt | model | StrOutputParser()
print(chain.invoke({"message": "When will the money go back on my card?"}))
```

### Give the next program named fields {#structured-output}

A title is useful to a person. If the next step needs `category`, put the valid choices in a schema. Below, `Literal` specifies three allowed values and Pydantic validates the parsed shape. Run `python langchain_basics.py classify` to see the script's example. The sentence about money could produce `refund_status`, but a valid category can still be the wrong one. Label a few ambiguous requests before using the result to route work. Handle model refusal, parsing errors, and timeouts at the application boundary. With `include_raw=True`, the structured-output wrapper exposes `raw`, `parsed`, and `parsing_error`. [Structured model output](https://docs.langchain.com/oss/python/langchain/models#structured-output) · [`ChatOpenAI.with_structured_output` reference](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output)

```python
# Continue with the same model.
from typing import Literal

from pydantic import BaseModel

class TicketLabel(BaseModel):
    category: Literal["return", "refund_status", "other"]
    summary: str

extract = model.with_structured_output(TicketLabel, method="json_schema")
label = extract.invoke("Classify and summarize: when does the refund reach my card?")
print(label.model_dump())
```

Structured output gives the caller fields to read. A tool call asks the runtime to execute an operation with arguments. An agent can do both. If you configure `create_agent(response_format=TicketLabel, ...)`, its final parsed result is available under `structured_response`. Some models use a function-call protocol to submit that schema. Business tools still run through the application's tool layer. [Agent structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

### Follow the policy lookup across messages {#tool-call}

`lookup_policy` is a Python function with a typed argument and a docstring. `@tool` exposes them in the tool description. Start by running `.invoke()` on the tool yourself. Then let the agent see the same tool and inspect the messages. This example continues with the model from above. The `agent` command in `langchain_basics.py` prints the tool requests, tool messages, and final answer. [Defining a LangChain tool](https://docs.langchain.com/oss/python/langchain/tools) · [Agents](https://docs.langchain.com/oss/python/langchain/agents)

```python
from typing import Literal

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

@tool
def lookup_policy(topic: Literal["return_window", "damaged_item"]) -> str:
    """Read one policy from the demo shop's handbook."""
    policies = {
        "return_window": "[R1] Unused items may be returned within 30 days of delivery.",
        "damaged_item": "[R2] Contact support within 7 days and attach photos.",
    }
    return policies[topic]

print(lookup_policy.invoke({"topic": "return_window"}))
agent = create_agent(
    model=model,
    tools=[lookup_policy],
    system_prompt="Look up the policy before answering. Cite its policy ID.",
)
result = agent.invoke(
    {"messages": [HumanMessage("How long do I have to return an unused item?")]},
    config={"recursion_limit": 12},
)

for message in result["messages"]:
    print(message.type, message)
```

In a typical trace you see `human → ai → tool → ai`. The first AI message can have empty text and a populated `tool_calls` list. For a request with the illustrative ID `call_demo`, the runtime calls the tool with `return_window` and attaches the result to a `ToolMessage` with `tool_call_id="call_demo"`. The next model call sees both the request and the result. A model may request multiple tools, take another turn, or answer without making a call. Inspect the trace before assuming it read the policy. If every policy answer must use the handbook, run the lookup as a required step in your workflow. The graph's recursion limit bounds execution steps. Production code can also set an explicit model-call or tool-call limit with middleware. [Tool calling with models](https://docs.langchain.com/oss/python/langchain/models#tool-calling) · [Tool messages](https://docs.langchain.com/oss/python/langchain/messages#tool-message) · [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

![The model requests lookup_policy with ID call_demo. Python runs the lookup and sends a ToolMessage with the same ID and policy R1. The model can then cite R1.](tool-loop.svg "Figure 1. The model asks for a lookup. The application runs the function and pairs its result with the request ID.")

## Route a ticket with an offline graph {#graph}

Run `python ticket_graph.py`. The script exercises four tickets with plain Python functions. `triage` reads a message and writes a `route`. The refund branch gets its amount from a mock order record for `A-104`, writes a draft for 2,400 cents, then stops at review. Shipping returns a tracking help status. A refund for an unknown order goes to manual review. The program records only draft decisions and tracking statuses.

![Ticket triage sends a known refund to draft and review, a shipping question to tracking help, and an unknown order or request to manual review.](ticket-routes.svg "Figure 2. One conditional edge chooses a branch. The refund branch prepares a draft and waits for review.")

- **State** is the data passed between steps. `Ticket` declares `order_id`, `request`, the eventual route, and fields such as `refund_cents` and `audit`.
- **Nodes** read state and return just the changed fields. `draft_refund`, for example, reads the mock order and returns the amount and one audit entry.
- **Edges** schedule work. `START → triage` is fixed. The conditional edge out of `triage` reads its updated `route`. `END` marks a terminal path. `compile()` produces the object you can `invoke` or `stream`. [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

Here is the wiring for the example. The full file supplies `Ticket`, the five node functions, and the imports.

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

The default update for a state field replaces its previous value. The audit field uses `audit: Annotated[list[str], add]`, with `add` from `operator`. Each node returns a new entry such as `{"audit": ["draft"]}`. The reducer joins the old list and that entry. Returning the entire old list plus a new item would append the history a second time. Parallel nodes writing the same ordinary field in one super-step, a round of graph execution, need a reducer or distinct output fields. If order matters, include an explicit sort key and sort after collection. For conversation state, `MessagesState` uses `add_messages`, which can update an existing message by ID. [State and reducers](https://docs.langchain.com/oss/python/langgraph/graph-api-state) · [Concurrent update error](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE)

| `ticket_graph.py` case | What the assertions check |
|---|---|
| `A-104` refund, approval | Pauses at `review` with 2,400 cents. Resumes to `draft_approved` with three audit entries |
| `A-104` refund, rejection | Same pause. Resumes to `draft_rejected` |
| `想查物流` (check shipping) | Returns `show_tracking_help` without a pause |
| Unknown order, refund request | Returns `needs_manual_review` without a pause |

For an extension, add a damaged-item route. Assert the chosen terminal status first. Then try replacing the keyword classifier with a typed model call. Keep order existence, amount, and eligibility checks in application code. When the work expands to a variable number of attachments, [`Send`](https://docs.langchain.com/oss/python/langgraph/graph-api-send-command) can dispatch one worker per attachment and a reducer can collect the results.

## Save progress and resume the same ticket {#persistence}

The **checkpointer** records snapshots at graph execution boundaries. A **thread** identifies the workflow history you want to use. Pass the same `thread_id` when a supervisor responds. Give a separate ticket a separate ID so inputs and append-only fields stay with the intended workflow. `InMemorySaver` holds snapshots while the process lives. For a single-machine exercise across restarts, try a SQLite checkpointer. For a shared service, evaluate a database-backed checkpointer such as PostgreSQL. A **store** holds data an application wants to retrieve across threads, for example the same staff member's language preference. Store access should be scoped by account or tenant. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers) · [Stores](https://docs.langchain.com/oss/python/langgraph/stores)

Inside `review`, `interrupt(payload)` hands the caller the order and amount. A later `Command(resume=True)` makes `True` the return value of `interrupt()` in that node. Try this block from a Python terminal in the lab directory. Keep both calls in the same process for the in-memory checkpointer.

```python
from langgraph.types import Command
from ticket_graph import graph

config = {"configurable": {"thread_id": "try-a-104"}}
paused = graph.invoke(
    {"order_id": "A-104", "request": "我想申请退款", "audit": []},
    config,
)
print(paused["__interrupt__"][0].value)  # pending review payload
print(graph.get_state(config).next)     # ('review',)
final = graph.invoke(Command(resume=True), config)
print(final["status"], final["audit"])
# draft_approved ['triage:refund', 'draft', 'draft_approved']
```

![Thread try-a-104 checkpoints after preparing the refund draft. Review pauses for a decision, then restarts from the beginning of its node on the same thread. The result is a draft decision.](checkpoint-resume.svg "Figure 3. Resume uses the same thread. The review node starts again and receives the supervisor's decision at interrupt().")

**Re-execution matters when a node can change the outside world.** The interrupted `review` node starts again at its first line. Keep draft preparation in an earlier node and any real refund in a separately authorized step after review. Bind approval to the exact draft version and check the approver's identity, role, and access to the ticket on the server. If the payment processor accepts the refund and its response is lost, a retry should reuse the same operation key, such as the approved refund ID. Record the external result and reconcile uncertain requests with the payment system. Checkpoint persistence covers graph state. Idempotency for an external operation depends on that operation's implementation and the system it writes to. [Interrupt rules](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts) · [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

Use `get_state(config).values` to inspect a snapshot, `.next` to see pending nodes, and `get_state_history(config)` to inspect earlier checkpoints. Replaying from an earlier checkpoint can re-run downstream steps. Use synthetic data or read-only tools when investigating that path. With a network operation, limit retries to errors where retry is appropriate. For a loop, budget attempts in state and keep a terminal path for exhaustion. [Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) · [Graph recursion limit](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)

### See what each node returned {#stream}

Run the following in the same Python session, using a separate thread for the shipping example. With `version="v2"`, `updates` events carry the node's changed fields in `data`. Use `values` when you want state snapshots. `messages` is useful in a model-backed graph to stream model chunks, and `custom` lets your code emit progress such as “searching the handbook.” Send users only the events and fields they can see. [Streaming modes](https://docs.langchain.com/oss/python/langgraph/streaming)

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

You should see the `triage` update followed by the `shipping` update. In an approval flow, expose a waiting status for the reviewer and a separate resume request. A web connection can close while the checkpoint remains available.

## Build a knowledge base that cites its evidence {#rag}

The shop's internal help center, or knowledge base, has an employee article on two-factor authentication (2FA) and a restricted administrator article on single sign-on (SSO). A question such as “I switched phones. How do I restore 2FA?” should find the employee page even when the first search misses. An employee asking about the administrator-only procedure should receive a response that the available material cannot answer it.

This is an application of **retrieval-augmented generation (RAG)**. Start with a fixed retrieval step followed by an answer step. For a real handbook, index useful passages instead of whole manuals. Keep a stable chunk ID, document title, source link, version, and visibility scope with each passage. At request time, the retriever returns permitted passages for the question. Assemble that context for the model, and return references to the passages it used. A dense vector index searches by semantic similarity. Keyword search is still useful for exact error codes and policy identifiers. [LangChain semantic search and minimal RAG tutorial](https://docs.langchain.com/oss/python/langchain/knowledge-base)

Now run `python offline_rag.py` to see a version that can search a second time. The lab uses two in-memory `Document` objects and deterministic string matching. A dictionary rewrites one known alias, and the answer step copies the authorized source text. You can trace every decision before adding an embedding service or a model.

![The retriever filters passages by scope. A hit produces an answer with source IDs. A miss can be rewritten once and searched again, or end in abstention.](rag-loop.svg "Figure 4. The second search has a one-rewrite budget. Both searches enforce document scope before any passage reaches the answer step.")

| Lab input under `scope="employee"` | Searches | Result |
|---|---:|---|
| `重置双重验证` (reset 2FA) | 1 | Answer with `KB-17#1` |
| `换手机了，怎么恢复 2FA？` (new phone) | 2 | Rewrite and answer with `KB-17#1` |
| `重置单点登录` (reset SSO) | 1 | Abstain. The matching article is restricted |
| `能代我修改账户吗？` (change my account for me) | 1 | Abstain. The available articles do not support an answer |

The conditional edge after `retrieve` calls `next_step` to look for a hit or decide whether to try again. In the excerpt below, `MAX_ATTEMPTS = 2` includes the first search.

```python
def next_step(state: State):
    if state["retrieved_ids"]:
        return "answer"
    if state["attempts"] < MAX_ATTEMPTS and state["query"] in ALIASES:
        return "rewrite"
    return "abstain"
```

The second search applies the same scope check. On a final miss, `abstain` returns a short explanation and an empty citation list. The four assertions check counts, citations, and the abstention flag. [LangGraph custom RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

With a model key, run `python langchain_basics.py rag` to add a generation step to the same retrieval lab. The script sends the original question and only the matched, permitted passages to the model. A successful reply has an `answer` and `citations: ["KB-17#1"]`. The script checks that the answer and citations are present and that every returned ID was in its context. Check the cited sentence yourself to see whether it supports the answer.

### Adapt the lab to a real document collection {#rag-upgrade}

1. **Keep a baseline.** Implement one retrieval followed by generation. Keep the original question separate from `search_query` so a rewritten query can improve retrieval while the answer still addresses the user's request. Compare results on the same questions before adding an extra model call for rewriting.
2. **Enforce authorization before passage text reaches the model.** In the lab, `scope` is a fixture. In a service, get scope from the authenticated identity and enforce access inside the retriever. Retrieved documents may contain instructions aimed at the model. Treat them as evidence, restrict available tools, and test a document that tries to cause an unauthorized action. [OWASP on indirect prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
3. **Validate the answer and citations.** Have the model return `answer` and `citations`. Require at least one citation for factual answers. Check every cited ID against the permitted passages actually supplied to the model. Build links from trusted source metadata. A valid ID can still point to text that fails to support the claim. Add a separate evidence-support check and a human route when approved sources disagree or leave an important gap.

## Test the paths and measure answer quality {#evaluation}

Start with function tests for the tool's accepted arguments and the graph's routing rules. Use a fake chat model to produce a predetermined tool request, then assert that the graph executes the tool and matches its result to the request ID. Test a closed graph path, a pause followed by approval, a pause followed by rejection, and a request outside the caller's scope. Where a real write exists, simulate an external service that accepts the request then times out and test the deduplication behavior. [LangChain unit tests](https://docs.langchain.com/oss/python/langchain/test/unit-testing) · [LangGraph testing guide](https://docs.langchain.com/oss/python/langgraph/test)

For retrieval quality, make a fixed dataset with `question`, the caller's role, expected source IDs, a reference answer or scoring rule, and whether the system should abstain. Include direct questions, paraphrases, answers requiring more than one passage, conflicting versions, permission boundaries, and unknown topics. Keep the prompt, model, and index versions alongside each run. [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) · [RAG evaluation tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)

| Measure | How to count it |
|---|---|
| Hit@k | Among answerable questions with labeled sources, fraction that retrieved at least one labeled source in the top `k` |
| Recall@k | On the same questions, average the fraction of each question's labeled sources present in the top `k` |
| Citation ID validity | Valid IDs actually in the permitted model context divided by all returned citation IDs. Count missing citations on factual answers separately |
| Evidence support | Supported factual claims divided by all checkable factual claims. Use human grading or calibrate an automated grader against people |
| Abstention recall | Correct abstentions divided by questions that should be declined |
| False abstention rate | Abstentions on answerable questions divided by all answerable questions |

For example, if the answer needs sources A and B and the retriever finds only A, that question counts as a hit and its recall is `1/2`. Choose a consistent top `k` cut for routed retrieval, such as the final passages the model sees. Report the counts and mark an empty denominator as unavailable. Compare one search with two allowed searches while holding the dataset and other components fixed. Record the median and 95th-percentile latency, tool and model call counts, and input/output tokens so you can see the cost of recovering additional hits.

In a service, a trace is the sequence of steps for one request. Start with node names, timings, permitted source identifiers, counts, and routing outcomes. Review data collection and redaction before sending real traces to a service such as LangSmith. Inspect representative failures and bring approved, redacted examples back into the offline dataset. [LangChain tracing](https://docs.langchain.com/langsmith/trace-with-langchain) · [Sensitive data in traces](https://docs.langchain.com/langsmith/mask-inputs-outputs)

| What you observe | What to check first |
|---|---|
| `ModuleNotFoundError` or an obsolete import | Verify the active interpreter, `python -m pip check`, and the v1 migration guide. Avoid local files named `langchain.py` or `langgraph.py` |
| Tool fails or the answer invents a clause | Inspect the tool request ID and arguments, run the tool directly, and read the returned `ToolMessage` |
| Resume goes to the wrong ticket or finds no checkpoint | Check the exact `thread_id`, the checkpointer's lifetime, and whether the snapshot has a pending review |
| The graph exceeds its recursion limit | Log the route and attempt counter and find the branch that should lead to `END` |
| A relevant document goes missing | Look up its chunk ID and scope, then inspect the query, version filters, and ranked candidates |
| The source was found but the answer is wrong | Read the passages the model actually received and verify whether they support the cited claim |

## Learning resources and next exercises {#resources}

1. **Start with model calls and tools.** Run the Python course sections on models and messages, tools, and structured output. Create a small tool that accepts one argument and inspect the corresponding `ToolMessage`. {{< new-tab "LangChain Essentials (Python)" "https://academy.langchain.com/courses/langchain-essentials-python" >}}
2. **Build and resume a graph.** Work through the nodes, edges, state, and interrupt lessons. Then use the 101 notebooks and inspect the email triage source for a larger example. {{< new-tab "LangGraph Essentials (Python)" "https://academy.langchain.com/courses/langgraph-essentials-python" >}} · {{< new-tab "LangGraph 101 notebooks" "https://github.com/langchain-ai/langgraph-101" >}} · {{< new-tab "Email agent source" "https://github.com/langchain-ai/langgraph-101/blob/main/agents/email_agent/graph.py" >}}
3. **Build a cited search application.** Index a small public handbook and verify the source for five questions. Try one bounded rewrite, then compare the resulting source hits and false abstentions. {{< new-tab "Semantic search and basic RAG" "https://docs.langchain.com/oss/python/langchain/knowledge-base" >}} · {{< new-tab "Custom RAG with LangGraph" "https://docs.langchain.com/oss/python/langgraph/agentic-rag" >}} · {{< new-tab "RAG evaluation with LangSmith" "https://docs.langchain.com/langsmith/evaluate-rag-tutorial" >}}
4. **Study control-flow patterns.** Read the diagrams for chaining, routing, and parallel work. For a read-only research assistant, start with one agent and a call budget. Try separate workers once you have a concrete need for different tools, permissions, or parallel tasks. {{< new-tab "Building effective agents" "https://www.anthropic.com/research/building-effective-agents" >}} · {{< new-tab "ReAct paper" "https://arxiv.org/abs/2210.03629" >}} · {{< new-tab "LangGraph workflow patterns" "https://docs.langchain.com/oss/python/langgraph/workflows-agents" >}}
