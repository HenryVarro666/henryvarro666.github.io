#!/usr/bin/env python3
"""Rebuild the bilingual figures for the LangChain and LangGraph note."""

from pathlib import Path

from precision_vs_recall_figs import Canvas


OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "langchain-langgraph-guide"


def card(c, x, y, w, h, title, lines, cls="f-node", title_size=17):
    c.rect(x, y, w, h, cls)
    c.text(x + 14, y + 25, title, title_size, weight=650, box=w - 28)
    for i, line in enumerate(lines):
        c.text(x + 14, y + 48 + i * 19, line, 14, box=w - 28)


def tool_loop(t):
    c = Canvas("tool-loop", 720, 406, t(
        "The application sends a user message. The model proposes lookup_policy with an ID. "
        "The application runs the function and sends back R1 in a ToolMessage with the same ID. "
        "The model can then answer with the policy citation.",
        "应用发送用户消息。模型请求调用 lookup_policy，并给出调用 ID。应用运行函数，"
        "把条款 R1 放到同一调用 ID 的 ToolMessage 中。模型据此生成带条款编号的回答。",
    ))
    c.text(24, 38, t("Follow one tool call", "跟踪一次工具调用"), 23, weight=650)
    c.text(160, 79, t("Chat model", "聊天模型"), 14, "f-m", "middle", 600)
    c.text(560, 79, t("Application", "应用程序"), 14, "f-m", "middle", 600)
    c.line(24, 91, 296, 91)
    c.line(424, 91, 696, 91)

    card(c, 424, 106, 272, 64, t("User asks about returns", "用户询问退货条件"),
         ("HumanMessage",))
    card(c, 24, 106, 272, 64, t("Request a lookup", "请求查询条款"),
         ("lookup_policy(return_window)",))
    c.arrow(419, 139, 301, 139)
    c.text(360, 126, "messages", 13.5, "f-m", "middle")

    c.path([(160, 175), (160, 238), (225, 238)], "f-ln", 1.5)
    c.arrow(225, 238, 419, 238)
    c.text(291, 209, "AIMessage.tool_calls", 14, "f-t", "middle", 600)
    c.text(291, 227, "id = call_demo", 13.5, "f-m", "middle")
    card(c, 424, 207, 272, 64, t("Run the Python function", "执行 Python 函数"),
         (t("Read [R1] · 30 days after delivery", "读取 [R1] · 送达后 30 天"),), "f-hl")

    c.path([(560, 276), (560, 347), (495, 347)], "f-ln", 1.5)
    c.arrow(495, 347, 301, 347)
    c.text(430, 316, "ToolMessage", 14, "f-t", "middle", 600)
    c.text(430, 334, "tool_call_id = call_demo", 13.5, "f-m", "middle")
    card(c, 24, 314, 272, 64, t("Answer with [R1]", "回答时引用 [R1]"),
         ("AIMessage",), "f-hl")
    return c


def ticket_routes(t):
    c = Canvas("ticket-routes", 720, 513, t(
        "START leads to triage. A known refund request leads to draft_refund and then review, "
        "where the graph pauses. A tracking question goes to shipping. Unknown requests and "
        "unknown refund orders go to manual. Each completed path reaches END.",
        "START 进入 triage。已知订单的退款请求先生成草稿，再暂停等待审核。物流问题进入 shipping，"
        "未知问题或查不到订单的退款请求进入 manual。每条路径处理完都到达 END。",
    ))
    c.text(24, 38, t("A ticket takes one of three paths", "一张工单走三条分支中的一条"), 23, weight=650)
    c.rect(292, 58, 136, 34, "f-node", rx=17)
    c.text(360, 80, "START", 14, "f-t", "middle", 650)
    c.arrow(360, 96, 360, 115)
    card(c, 260, 120, 200, 62, "triage", (t("Classify the request", "按请求类型分流"),), "f-hl")

    c.path([(360, 186), (360, 199), (130, 199), (130, 204)], "f-ln", 1.5)
    c.path([(360, 199), (590, 199), (590, 204)], "f-ln", 1.5)
    c.line(360, 199, 360, 204, "f-ln", 1.5)
    for x, branch in ((130, "refund"), (360, "shipping"), (590, "manual")):
        c.text(x, 221, branch, 14, "f-t", "middle", 600)
        c.arrow(x, 228, x, 243)

    card(c, 32, 249, 196, 63, "draft_refund", (t("Look up the amount", "查询金额并生成草稿"),), "f-hl")
    card(c, 262, 249, 196, 63, "shipping", (t("Show tracking help", "展示物流查询指引"),))
    card(c, 492, 249, 196, 63, "manual", (t("Send for manual review", "交给客服人工处理"),))
    c.arrow(130, 317, 130, 341)
    card(c, 32, 347, 196, 63, "review", (t("Wait for a decision", "暂停并等待人工决定"),), "f-altsoft")

    c.path([(130, 415), (130, 433), (360, 433)], "f-ln", 1.5)
    c.path([(590, 317), (590, 433), (360, 433)], "f-ln", 1.5)
    c.line(360, 317, 360, 433, "f-ln", 1.5)
    c.arrow(360, 433, 360, 451)
    c.rect(292, 456, 136, 34, "f-node", rx=17)
    c.text(360, 478, "END", 14, "f-t", "middle", 650)
    return c


def checkpoint_resume(t):
    c = Canvas("checkpoint-resume", 720, 326, t(
        "The graph creates a draft and pauses inside review, with review saved as the next node. "
        "A call with Command resume and the same thread ID restarts review from its first line. "
        "The interrupt then returns the reviewer's decision and the graph records the draft status.",
        "图生成草稿后在 review 内暂停，并保存下一待执行节点。用 Command resume 和同一 thread ID "
        "恢复时，review 从第一行开始运行。interrupt 返回审核人的决定，图记录草稿状态。",
    ))
    c.text(24, 38, t("What runs again after approval?", "提交审批结果后，哪段代码会重跑？"), 23, weight=650)

    c.rect(24, 63, 672, 42, "f-node")
    c.text(360, 90, t("Both calls use the same thread_id", "两次调用使用同一个 thread_id"),
           16, "f-t", "middle", 600)
    card(c, 24, 133, 196, 88, t("First call", "第一次调用"),
         ("draft_refund → review", "interrupt(payload)"))
    card(c, 262, 133, 196, 88, t("Checkpoint", "保存检查点"),
         ('next = ("review",)', t("Wait for a decision", "等待人工决定")), "f-altsoft")
    card(c, 500, 133, 196, 88, t("Resume the run", "恢复执行"),
         ("Command(resume=…)", t("Restart review", "从 review 第一行执行")), "f-hl")
    c.arrow(225, 177, 256, 177)
    c.arrow(463, 177, 494, 177)

    c.arrow(598, 226, 598, 248)
    c.rect(24, 253, 672, 49, "f-node")
    c.text(360, 283, t("Save the outcome: draft_approved or draft_rejected",
                       "保存审核结果：draft_approved 或 draft_rejected"),
           16, "f-t", "middle", 600)
    return c


def rag_loop(t):
    c = Canvas("rag-loop", 720, 414, t(
        "Retrieve using the question and the authorized scope. A permitted match leads to an answer with a source ID. "
        "A first miss with a known alias is rewritten and retried once. Other misses lead to abstention.",
        "用问题和授权范围检索。找到可见证据就带来源编号回答。首次未命中且有已知同义问法时，"
        "改写并重查一次。其余未命中情况都拒答。",
    ))
    c.text(24, 38, t("Retrieval gets at most two attempts", "最多检索两次"), 23, weight=650)
    c.rect(224, 59, 272, 43, "f-node")
    c.text(360, 86, t("Question + authorized scope", "用户问题与本次授权范围"),
           16, "f-t", "middle", 600)
    c.arrow(360, 106, 360, 128)
    card(c, 237, 133, 246, 63, "retrieve", (t("Filter by scope and match", "先按范围过滤，再匹配问题"),), "f-hl")

    c.path([(360, 201), (360, 219), (121, 219), (121, 226)], "f-ln", 1.5)
    c.path([(360, 219), (590, 219), (590, 226)], "f-ln", 1.5)
    c.line(360, 219, 360, 226, "f-ln", 1.5)
    for x, label in ((121, t("Authorized hit", "命中可见证据")),
                     (360, t("Alias on first miss", "首次未命中且有同义问法")),
                     (590, t("Other misses", "其余未命中情况"))):
        c.text(x, 244, label, 14, "f-t", "middle", 600)
        c.arrow(x, 251, x, 269)

    card(c, 24, 274, 194, 62, "answer", (t("Cite the source ID", "回答并附上来源编号"),), "f-hl")
    card(c, 262, 274, 196, 62, "rewrite", (t("Use the known alias", "用已知同义问法改写"),))
    card(c, 492, 274, 196, 62, "abstain", (t("Report missing evidence", "说明现有证据不足"),), "f-altsoft")
    c.path([(360, 341), (360, 384), (704, 384), (704, 164), (643, 164)], "f-ln", 1.5)
    c.arrow(643, 164, 488, 164)
    c.text(514, 373, t("Retry once", "重查一次"), 14, "f-m", "middle", 600)
    return c


def english(en, _zh):
    return en


def chinese(_en, zh):
    return zh


def main():
    for translate, suffix in ((english, ".svg"), (chinese, ".zh.svg")):
        for draw in (tool_loop, ticket_routes, checkpoint_resume, rag_loop):
            canvas = draw(translate)
            if canvas.warnings:
                raise ValueError("\n".join(canvas.warnings))
            (OUT / (canvas.name + suffix)).write_text(canvas.svg(), encoding="utf-8")


if __name__ == "__main__":
    main()
