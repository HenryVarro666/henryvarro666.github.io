---
title: "dbt 学习笔记：从第一条 SQL 模型到可上线的订单日报"
date: 2026-11-13T19:46:15-06:00
description: "用 dbt v2 和本地 DuckDB 做一个完整订单实验，练习分层建模、数据测试、迟到数据处理、快照与生产发布。"
topics: [data-science, ml-systems]
---

一张订单表有四笔订单，合计 116 美元。把它直接关联到支付投递表，再按订单金额求和，结果变成了 166 美元。数据库没有报错：拆分支付和重送让同一笔订单出现了三次。这个错误很适合用来学 dbt，因为修复得从两个问题下手：**一行代表什么？下游可以依赖哪些条件？**

这份笔记适合能读懂结构化查询语言（SQL）里的 `SELECT`、`JOIN` 和 `GROUP BY`，第一次用 dbt 的读者。预留 45–60 分钟可以跑完主实验。只想动手的话，从[本地安装](#setup)开始；有项目经验的话，可以直接看[迟到数据与重跑](#incremental)或[生产发布](#production)。

## dbt 在数据管道的什么位置？ {#mental-model}

dbt 管理的是**转换代码和模型之间的依赖**。一个 SQL 模型通常是一条带名字的 `SELECT`。你把查询放进 Git，用 `ref()` 表达它引用哪个模型，用测试写明结果应该满足什么条件。dbt 解析项目、展开模板、排好依赖顺序，再把实际 SQL 交给 DuckDB、Snowflake、BigQuery 或 Databricks 等目标执行。

![采集工具将输入写到原始表，dbt 编译并提交转换 SQL，数据引擎建出供报表读取的模型。](elt-boundary.svg "图 1：在 ELT 流程中，采集工具完成 E 和 L，dbt 管理 T；查询由目标数据引擎执行。")

**提取、加载、转换（ELT）**把原始数据先装入目标，再就地转换。提取、转换、加载（ETL）会在装入目标之前先转换。常见分工是：连接器或采集任务负责搬数据，dbt 把原始表变成可用的数据集，调度器负责在输入到齐后启动任务，商业智能（BI）工具负责报表。dbt 的 seed 可以加载小型 CSV 文件，这适合固定字典和练习数据；业务数据库的定期同步和应用程序接口（API）数据采集仍应由上游任务完成。

### v2、dbt OSS 和旧教程如何对应 {#versions}

| 阅读材料里写的名字 | 今天怎么理解 | 学习时怎么选 |
|---|---|---|
| **dbt v2**，早期材料中称 Fusion | 当前一代 Rust 引擎的主要发行版。可以本地使用，部分功能需要登录 dbt platform。 | 本实验使用 `dbt==2.0.11`，本地适配器使用内置的 DuckDB。 |
| **dbt OSS**，早期材料中称 dbt Core v2 | 同一代框架的 Apache 2.0 开源发行版。 | 有开源许可要求时，查官方安装说明及功能表。 |
| **dbt v1**，常见旧称 dbt Core v1 | Python 引擎，旧教程常用 `dbt-core` 加一个独立适配器包。 | 维护 v1 项目时另开环境，按团队锁定的引擎和适配器版本运行。 |
| **dbt platform**，旧资料常说 dbt Cloud | 托管的开发及运行环境，可提供编辑器、作业、目录等能力。 | 想练团队协作和托管作业时再用；本实验不需要账号。 |

版本和适配器状态按 **2026-11-13** 核对。复制教程之前先看 `dbt --version`；同一个命令名可能来自不同版本的可执行文件。尤其要核对测试参数的嵌套方式、目标适配器支持的增量策略，以及文档命令产出的文件。[官方 v2 导览](https://docs.getdbt.com/docs/introduction?version=2) · [发行版区别](https://docs.getdbt.com/blog/comparing-dbt-and-dbt-oss) · [数据平台支持表](https://docs.getdbt.com/docs/supported-data-platforms?version=2)

## 本地安装：先把项目跑起来 {#setup}

[下载完整订单实验（ZIP）](/notes/dbt-learning-guide/dbt-orders-lab.zip)。解压后进入 `dbt-orders-lab` 目录。数据都是虚构的，实验只写本地的 `orders_lab.duckdb` 文件。下面以 macOS/Linux 和 Python 3.12 为例，版本固定是为了便于复现。

```bash
cd dbt-orders-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
dbt --version

python load_data.py initial
dbt debug --profiles-dir .
dbt build --profiles-dir .
```

Windows PowerShell 可用 `py -3.12 -m venv .venv` 建环境，运行 `.\.venv\Scripts\Activate.ps1` 激活，再执行余下的 Python 和 dbt 命令。`dbt --version` 应显示 **2.0.11**。首次构建应显示 **26 个任务成功**：6 个模型、17 个数据测试、1 个单元测试、1 个 seed 和 1 个快照。本地验证使用了 Python 3.12、dbt 2.0.11；加载脚本用 `duckdb==1.5.0` 写入同一个数据库文件。DuckDB 的 dbt v2 适配器已经内置。[安装 dbt](https://docs.getdbt.com/docs/local/install-dbt) · [v2 的 DuckDB 配置](https://docs.getdbt.com/docs/local/connect-data-platform/duckdb-setup)

```text
dbt-orders-lab/
├── dbt_project.yml                 # 项目名、模型路径、各层的默认构建方式
├── profiles.yml                    # 连接配置：DuckDB 文件、目标 schema
├── data/                           # 模拟输入和后续批次
├── load_data.py                     # 模拟上游把数据写入 raw
├── seeds/payment_statuses.csv       # 小型、随版本维护的状态字典
├── models/
│   ├── staging/                    # 认领原始表，统一字段和类型
│   ├── intermediate/               # 选支付版本，按订单聚合
│   ├── marts/                      # 订单事实表、按天按渠道的日报
│   └── _properties.yml             # 说明、数据测试、单元测试
├── tests/no_order_overcapture.sql
└── snapshots/orders_history.yml
```

`dbt_project.yml` 中的 `profile: orders_lab` 指向下方的配置。YAML 是这里用来写配置的文本格式。`--profiles-dir .` 指定从当前目录找 `profiles.yml`，不借用其他项目的连接；这个 profile 指向本地文件。一个 **target** 是一套可选的连接和输出位置，`dev` 目标把模型写进 `analytics_dev`，输入仍然在 `raw`。从实验根目录执行命令，因为连接使用相对路径。[连接配置参考](https://docs.getdbt.com/docs/local/profiles.yml)

```yaml
# profiles.yml
orders_lab:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: orders_lab.duckdb
      schema: analytics_dev
      threads: 2
```

> **先判定卡在哪。** `dbt --version` 不对，先检查虚拟环境和 `dbt` 的路径。`debug` 通过却找不到 `raw.orders` 时，确认执行过 `load_data.py initial`，并且 Python 和 dbt 在读写同一个文件。换一个全新目录解压即可从头开始；重跑 `initial` 只替换输入表，已有的模型和快照历史还在。

## 从两张原始表建出日报 {#lab}

这个模拟店铺的金额用整数美分表示，币种统一为美元。**下单金额**来自订单，**实收**只计当前版本中状态为 `captured` 的支付。`paid_order_count` 表示至少有正额实收的订单数，部分付款也算。实收按**下单日**归属，所以它与按支付发生日统计的现金流报表会不同。实验没有退款；真实系统需要单独定义退款、拒付、汇率及金额口径。

![支付和订单从原始表进入 staging；支付先定版本、再按订单聚合，与订单汇合成事实表，最后生成日报。](orders-lineage.svg "图 2：依赖箭头从输入指向使用者。订单粒度的事实表是两路数据汇合的位置；快照和测试未画在主路径上。")

### 用 source() 和 ref() 声明依赖 {#sources}

`source('shop', 'orders')` 找的是 **dbt 外部已存在**的表。`shop` 是项目中的逻辑名，`schema: raw` 决定去哪里找；加上这段声明不会创建或加载 `raw.orders`。项目自己的模型、seed 和快照用 `ref()` 引用。编译时，dbt 把引用换成当前目标使用的关系名，同时登记依赖。它根据这个有向无环图（DAG）决定执行顺序，文件名的字母顺序不起作用。[source()](https://docs.getdbt.com/reference/dbt-jinja-functions/source) · [ref()](https://docs.getdbt.com/reference/dbt-jinja-functions/ref)

```yaml
# models/staging/_sources.yml（节选）
version: 2
sources:
  - name: shop
    schema: raw
    tables:
      - name: orders
      - name: payment_events
```

```sql
-- models/staging/stg_orders.sql
select
    order_id,
    ordered_on,
    lower(trim(channel)) as channel,
    order_total_cents
from {{ source('shop', 'orders') }}
```

双花括号内是 **Jinja** 模板表达式。查看编译后的 SQL，可以把模板问题和目标方言的问题分开查：

```bash
dbt compile --profiles-dir . --select fct_orders
# 查看 target/compiled/orders_lab/models/marts/fct_orders.sql
```

### 写 SQL 之前先写清粒度 {#grain}

| 表或模型 | 一行是什么 | 识别这一行的键 |
|---|---|---|
| `raw.orders` / `stg_orders` | 一笔订单 | `order_id` |
| `raw.payment_events` / `stg_payments` | 一次支付记录投递，可能是重送 | `delivery_id` |
| `int_payments_current` | 一笔支付的当前业务版本 | `payment_id` |
| `int_payments_by_order` / `fct_orders` | 一笔订单的支付汇总 / 一笔完整订单 | 各自都是 `order_id` |
| `mart_daily_orders` | 一个下单日和一个渠道的指标 | `(ordered_on, channel)` |

订单 `o100` 有两笔支付，其中一笔重送过。直接关联五条投递后，`o100` 的 2,500 美分出现三次；对所有订单求和多算了 5,000 美分。可以在初始批次后用以下查询复现：

```bash
dbt show --profiles-dir . --inline "
select count(*) as joined_rows, sum(o.order_total_cents) as wrong_cents
from {{ ref('stg_orders') }} as o
left join {{ ref('stg_payments') }} as p on o.order_id = p.order_id"
```

预期结果是 **6 行、16,600 美分**。先按 `payment_id` 选当前版本，再按 `order_id` 汇总支付，最后回到订单粒度，就能消除重复。用 `sum(distinct order_total_cents)` 会把碰巧同价的两笔订单合并，不能修复关联到错误粒度的问题。[Kimball：如何定义粒度](https://www.kimballgroup.com/2003/03/declaring-the-grain/)

### 分层及其物化方式 {#materializations}

**Staging** 通常按源系统组织，一份模型对应一张输入表。在这里做列名统一、类型转换和清理状态字符串。**Intermediate** 为具体任务服务，比如选当前版本和聚合到订单。**Marts** 由业务过程或消费需求组织：`fct_orders` 可用来对账，`mart_daily_orders` 可直接给报表取数。分层属于项目约定。一笔简单转换写成一个好懂的模型就够了。

| 物化方式 | dbt 的输出 | 第一次使用时怎么判断 |
|---|---|---|
| `view` | 一份保存的查询定义，读取时执行 | 轻量清洗可先用视图。链太深或被高频读取时测查询成本。 |
| `table` | 执行查询并保存结果；通常每次构建重算该模型 | 刷新可以承受、读取频繁的事实表和日报适合先用表。 |
| `incremental` | 首次建表，后续按模型条件和适配器策略写入变动 | 全量处理确实过慢才引入。先定义变动范围、匹配键和回填办法。 |
| `ephemeral` | 内联为下游查询里的公共表表达式（CTE） | 仅少量模型复用的简单逻辑；没有可单独查询的对象。 |

这个项目默认 staging 和 intermediate 用视图、marts 用表；支付当前态在模型中显式改为增量。其他物化方式及行为取决于适配器，例如物化视图由目标平台决定何时刷新。目录名与数据仓库 schema 也是两件事；若以后加自定义 schema，先确认生成规则保留开发环境的前缀。[项目结构指南](https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview) · [物化方式](https://docs.getdbt.com/docs/build/materializations)

事实表从订单出发使用左关联，能保留没有支付尝试的 `o103`，并将它的实收补成 0：

```sql
-- models/marts/fct_orders.sql
select
    o.order_id,
    o.ordered_on,
    o.channel,
    o.order_total_cents,
    coalesce(p.captured_cents, 0) as captured_cents
from {{ ref('stg_orders') }} as o
left join {{ ref('int_payments_by_order') }} as p on o.order_id = p.order_id
```

```bash
dbt show --profiles-dir . --inline "
select * from {{ ref('mart_daily_orders') }}
order by ordered_on, channel"
```

| 下单日 | 渠道 | 订单数 | 有实收订单数 | 下单（美分） | 实收（美分） |
|---|---|---:|---:|---:|---:|
| 2026-01-01 | web | 2 | 1 | 4,300 | 2,500 |
| 2026-01-02 | store | 1 | 1 | 3,200 | 3,200 |
| 2026-01-02 | web | 1 | 0 | 4,100 | 0 |

## 质量检查：让错误可复现 {#testing}

**数据测试**查询实际数据，问“哪些行违反规则？”普通测试在失败行数为零时通过。内置的 `not_null`、`unique`、`accepted_values`、`relationships` 可检查非空、唯一、允许值和外键覆盖。唯一性和非空要分别声明；关联测试也不能代替对目标键唯一性的检查。下面用状态字典验证每条清洗后的支付状态，v2 中测试参数写在 `arguments` 下。CSV 意为逗号分隔值；这个 seed 只有两行，业务可以在 Git 里评审它的变化。

```yaml
# models/_properties.yml（节选）
- name: payment_status
  data_tests:
    - not_null
    - relationships:
        arguments:
          to: ref('payment_statuses')
          field: payment_status
```

需要跨列校验时可写独立 SQL 测试（singular test）。例如订单实收超过下单金额在这个店铺是不允许的，把这种行选出来就是测试。另一个手段是**单元测试**：给模型喂几行指定输入来验证分支。随附的单元测试给两笔订单只提供一笔支付汇总，断言第二笔订单还在、实收为 0。数据测试检查上线后真实数据有没有违约；单元测试可在合并前抓到有人把左关联改成内关联。[数据测试](https://docs.getdbt.com/docs/build/data-tests) · [单元测试](https://docs.getdbt.com/docs/build/unit-tests)

```sql
-- tests/no_order_overcapture.sql
select order_id, order_total_cents, captured_cents
from {{ ref('fct_orders') }}
where captured_cents < 0 or captured_cents > order_total_cents
```

```yaml
# models/_properties.yml (节选)
unit_tests:
  - name: includes_order_without_payment
    model: fct_orders
    given:
      - input: ref('stg_orders')
        rows:
          - {order_id: a, ordered_on: '2026-01-01', channel: web, order_total_cents: 1000}
          - {order_id: b, ordered_on: '2026-01-01', channel: web, order_total_cents: 500}
      - input: ref('int_payments_by_order')
        rows:
          - {order_id: a, captured_cents: 700}
    expect:
      rows:
        - {order_id: a, captured_cents: 700}
        - {order_id: b, captured_cents: 0}
```

```bash
dbt test --profiles-dir . --select 'test_type:unit'
```

先完成下一节的迟到批次，再动手注入坏数据。完整构建能够选中状态字典的关联测试；仅选择 `int_payments_current+` 不会把这项上游检查自动纳入。

## 迟到数据：找出变更，再选当前版本 {#incremental}

同一笔支付的投递可能晚几天才到，源系统也可能补一条旧版本。这里有两套时间：`event_at` 是业务发生时间，`loaded_at` 是这条投递实际入库的时间。它们服务于不同问题；而同一笔支付哪个版本有效，由源系统的 `revision` 决定。

![p02 于 1 月 1 日发生，1 月 4 日重送第 1 版，1 月 5 日收到更正后的第 2 版。](late-arrival.svg "图 3：更正的入库时间是 1 月 5 日。重算的是归属 1 月 1 日的订单指标。")

### 一次增量构建做两件事 {#incremental-sql}

首次运行时读全部投递。已经存在目标表并走增量时，先用入库时间窗口找到**变动的支付键**。随后读取这些支付的全部版本，按 `revision` 从高到低挑当前记录。在这批数据里，`p04` 的第 2 版在回看窗口外，迟到的第 1 版在窗口内。假如只对窗口内的行排序，就会把本来已实收的 3,200 美分改成 0。用 `merge` 按 `payment_id` 写入选出的记录；`unique_key` 只告诉 dbt 如何匹配，唯一性还要靠数据逻辑和测试。[增量模型](https://docs.getdbt.com/docs/build/incremental-models) · [适配器支持的策略](https://docs.getdbt.com/docs/build/incremental-strategy?version=2)

```sql
-- models/intermediate/int_payments_current.sql（节选）
{{ config(materialized='incremental', incremental_strategy='merge',
          unique_key='payment_id') }}

{% if is_incremental() %}
-- 在 changed_keys 中使用这个条件选择需要重算的键：
where loaded_at >= (
    select coalesce(max(loaded_at), timestamp '1900-01-01')
    from {{ this }}
) - interval '2 days'
{% endif %}
```

`{{ this }}` 代表当前模型对应的关系；首次建表或全量重建时不应查询尚不存在的增量目标。这里用当前态中最大的入库时间减两天，刻意保留重叠区间。这个例子假设输入是追加的、投递带可靠的入库时间和业务版本；当前态的最大时间不是严格的处理游标。生产上可以按已完成批次维护水位，设置回看窗口，并对超窗补数、物理删除、同版本内容冲突和失败重试分别定策略。若源事件很大，实测变动键筛选和历史回读是否真的减少扫描。

### 见证更正、重跑和恢复 {#changes}

```bash
python load_data.py late
dbt build --profiles-dir . --select 'int_payments_current+'
dbt show --profiles-dir . --inline "
select * from {{ ref('mart_daily_orders') }}
order by ordered_on, channel"
```

批次有三条：`p02` 的实收从 1,500 改成 1,200 美分；`p04` 晚到的旧版失败记录不能覆盖已有成功版本；`o101` 新到一笔 1,800 美分的实收。结果是 **1 月 1 日 web 的实收从 2,500 变为 4,000 美分，有实收订单从 1 笔变为 2 笔**。1 月 2 日 store 的 3,200 美分不变。再次加载 `late` 并运行同一构建，结果保持一致。SQL 脚本用 `delivery_id` 防重，模型用 `payment_id` 合并，两处处理的是不同的重放风险。

接着故意放入一个状态为 `unknown` 的投递：

```bash
python load_data.py bad
dbt build --profiles-dir .    # 本步预期失败
```

运行日志应报状态关联测试拦下一行，并显示 `mart_daily_orders` 被跳过。注意此时 `int_payments_current` **已经写入**，因为它自身不依赖状态 seed；多父节点测试会阻断依赖相关输入的下游，整次 `dbt build` 并不是跨所有表的原子事务。上一次成功的日报表也仍然存在。删除练习中注入的坏投递后，需要清理已经合入的旧状态；普通 `merge` 不会自动删除输入里消失的键：

```bash
python load_data.py fix
dbt build --profiles-dir . --select 'int_payments_current+' --full-refresh
```

应回到 **4,000 美分**。在生产使用 `--full-refresh` 之前，先核对原始历史是否齐全、重算会覆盖哪些关系、任务成本和读取方会受什么影响。大型表通常需要针对受影响的分区或业务键做回填。[build 与失败传播](https://docs.getdbt.com/reference/commands/build)

## 快照、宏和文档各管什么？ {#toolkit}

### 快照保留“当时是什么状态” {#snapshots}

如果源订单表会原地覆盖状态，日报只靠它不能回答“1 月 3 日这笔订单还是 open 吗？”**快照（snapshot）**给同一个业务键保留多行，记录每个版本的有效期，这种做法也称缓慢变化维度第 2 型（SCD Type 2）。项目以 `order_id` 为键，用可靠的 `updated_at` 检测状态变化。如果上游没有可靠的更新时间，再考虑显式比较列的 `check` 策略，并决定怎样处理源表中的删除。

先前的 `dbt build` 已记录了 `o100` 的第一版。现在更改输入，再取一次快照：

```bash
python load_data.py order-status
dbt snapshot --profiles-dir . --select orders_history
dbt show --profiles-dir . --inline "
select order_id, order_status, dbt_valid_from, dbt_valid_to
from {{ ref('orders_history') }}
where order_id = 'o100'
order by dbt_valid_from"
```

| order_id | 状态 | 从何时有效 | 何时结束 |
|---|---|---|---|
| o100 | open | 2026-01-01 09:00 | 2026-01-06 10:00 |
| o100 | complete | 2026-01-06 10:00 | `NULL`（当前） |

快照记录运行时能看到的变化。如果上午改了两次、作业晚上才跑，中间版本可能从未被看到。要求保留每次源端变更时，保留事件日志或考虑变更数据捕获（CDC）。快照保留观察到的历史，增量当前态维护每个键的一份最新结果。[快照](https://docs.getdbt.com/docs/build/snapshots) · [选增量还是快照](https://docs.getdbt.com/best-practices/how-we-handle-cdc/2-choosing-incremental-or-snapshots)

### 用宏复用小段逻辑，用文档对齐含义 {#macros-docs}

宏（macro）是可以复用的 Jinja 模板。假如多张 staging 表都要清理源状态，可在 `macros/` 中定义下面的宏，在模型里调用 `{{ clean_status('payment_status') }}`。先写出好读的 SQL，等含义和重复都明确后再抽取。跨仓库迁移时，先验证展开后的类型、日期和空值处理。

```sql
{% macro clean_status(column_name) -%}
  lower(trim({{ column_name }}))
{%- endmacro %}
```

第三方包也能提供宏与测试，例如 dbt-utils。采用前检查包与引擎、适配器的兼容性，声明版本并通过 `dbt deps` 安装。项目用了什么指标口径，写进模型及列的 `description`；对外共享的模型可以进一步加 **model contract** 固定字段和类型，另外用测试检查业务逻辑。需要多个报表按同一口径切片时，再评估语义层。公开数据目录里也应控制可见的字段和描述。[包与依赖](https://docs.getdbt.com/docs/build/packages) · [模型契约](https://docs.getdbt.com/docs/mesh/govern/model-contracts?version=2) · [语义层](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl)

```bash
dbt docs generate --profiles-dir .
dbt docs serve
# 在终端提示的本地地址查看模型、描述和血缘，Ctrl+C 退出。
```

dbt v2 的本地 Docs 用 Parquet 元数据生成静态页面；`manifest.json` 适合查模型依赖和做状态比较，`run_results.json` 适合查某一次运行的状态和耗时。处理报错时，结合实跑的日志及编译后 SQL。需要共享文档时，把产物当作内部数据资产管理：`target/` 也可能包含编译后的 SQL 和敏感的对象名。[文档生成](https://docs.getdbt.com/docs/build/documentation?version=2) · [产物说明](https://docs.getdbt.com/reference/artifacts/dbt-artifacts)

## 日常命令与排错顺序 {#commands}

下表省略了练习命令共用的 `--profiles-dir .`。命令里的 `+` 表示顺着依赖图选范围：前缀选上游，后缀选下游（都包含模型本身）。先用 `dbt ls` 看选中了什么，再发起会写数据的构建；查看选中范围时也留意间接选中的测试。

| 想做什么 | 命令 | 关注的结果 |
|---|---|---|
| 校验项目结构 | `dbt parse` | YAML、引用或项目配置出错时先修解析。 |
| 看将要提交的 SQL | `dbt compile --select fct_orders` | 打开 `target/compiled/`。编译不建模型，部分场景会读取目标元数据。 |
| 快速预览模型查询 | `dbt show --select mart_daily_orders --limit 5` | 预览查询定义的当前结果；如需确认已落地表的内容，用 `show --inline` 查询 `ref()`。 |
| 从头构建日报需要的上游 | `dbt ls --select '+mart_daily_orders'`，确认后换成 `dbt build` | 原始表必须已经加载；seed、模型和关联测试按选择执行。 |
| 看修改影响哪些下游 | `dbt ls --select 'int_payments_current+'` | 少跑无关分支；只跑子图时确认未重建的上游已经可用。 |
| 单独重建模型 | `dbt run --select fct_orders` | 只跑模型，不运行 seed、快照或测试；已有上游数据必须可读。 |
| 只跑已有模型的数据测试 | `dbt test --select 'test_type:data'` | 真实关系需要已构建；修改 SQL 后先确认检查的是新产物。 |
| 检查配置了阈值的输入是否新鲜 | `dbt freshness --select 'source:shop.payment_events'` | 适合做生产作业前置检查，需先配置阈值；静态练习数据不会随当前时间刷新。 |

| 看到的症状 | 先从哪里查 |
|---|---|
| `relation does not exist` | profile、目标 schema、源表是否加载、所选上游是否已构建。看编译后的完整表名。 |
| SQL 可以编译，但仓库拒绝执行 | 用目标仓库的方言复查函数、列类型和权限；检查连接和运行身份。 |
| 每次重跑总额都涨 | 在模型前后分别比较行数、键数和金额，检查追加语义、合并键和关联粒度。 |
| 图表仍显示昨天的值，测试全通过 | 读的是哪张表？何时加载、何时构建？数据结构正确不代表新批次到齐。 |

[命令参考](https://docs.getdbt.com/reference/dbt-commands) · [选择语法](https://docs.getdbt.com/reference/node-selection/graph-operators) · [show 行为](https://docs.getdbt.com/reference/commands/show)

## 从练习项目走到生产 {#production}

![开发者的个人 schema、隔离的 PR 检查、生产定时作业和对下游开放的报表表依次相连。](release-path.svg "图 4：开发环境、检查环境、生产环境使用不同的输出位置；先有失败处理规则，再给下游发布。")

1. **隔离输出和权限。** 开发者写自己有权访问的 schema；持续集成（CI）用临时的、按变更隔离的 schema，按需要读取生产副本或受控样本。生产作业用服务身份。连接密钥从环境或密钥管理服务注入；生产 profile 不复用本例的本地文件配置。
2. **给每次拉取请求（PR）安排构建与测试。** 小项目整库 `dbt build`，成本容易估计。较大项目可用 `dbt build --select 'state:modified+' --defer --state ./prod-state`。`./prod-state` 存放与生产相配的 manifest，单独于本轮的 `target/`；`--defer` 让未选中的部分上游可按规定引用生产关系。检查测试是否跨环境读数据、权限是否合适。这里的 manifest 比对是代码变更选择，不用于发现所有数据变化。
3. **把新鲜度和完整性放进调度。** 在真实 `payment_events` 的 source 配置里声明 `loaded_at_field: loaded_at`，再按数据生产节奏设 `warn_after` 和 `error_after`。调度器先等采集成功，再执行 dbt v2 的 `dbt freshness`，按告警规则决定是否继续构建。旧 v1 工作流常用 `dbt source freshness`。最大入库时间新鲜，只能证明到了某些新行；还要查批次清单、分区完整性或与源系统对账。
4. **决定输出何时对外可用。** 一个任务失败时，下游可能已写入一部分表。读者需要一致结果时，可先构建到候选 schema，在检查通过后按仓库能力切换稳定视图或发布指针。记录代码提交、输入批次、构建时间与失败的测试，约定响应人及回填步骤。点进一次成功的作业，应该能回答“日报用了哪一批数据”。

例如，一个每小时交付的源可以先用以下阈值做演练：两小时未收到数据时告警，四小时未收到数据时报错。把配置合并到真实项目的 source 声明；本实验的固定时间数据不适用这组阈值。

```yaml
sources:
  - name: shop
    schema: raw
    tables:
      - name: payment_events
        config:
          loaded_at_field: loaded_at
          freshness:
            warn_after: {count: 2, period: hour}
            error_after: {count: 4, period: hour}
```

[CI 官方指南](https://docs.getdbt.com/docs/deploy/continuous-integration) · [Defer](https://docs.getdbt.com/reference/node-selection/defer) · [Freshness 配置](https://docs.getdbt.com/reference/resource-configs/freshness) · [Freshness 命令](https://docs.getdbt.com/reference/commands/freshness)

### 把同一套做法用在什么地方？ {#applications}

| 应用 | 从哪张表开始 | 优先验证什么 |
|---|---|---|
| 电商日报或财务对账 | 一行一订单的事实表，另保留一行一支付的事实表 | 订单金额按下单日期，实收按事先约定的归属日期；退款按业务规则入账。对账时把币种和时区也列进定义。 |
| 产品漏斗或订阅留存 | 一行一事件的点击流，或一行一订阅状态版本 | 去重键、用户身份规则和时间边界。按报名批次算留存时，要排除还没有足够观察期的组。 |
| 机器学习（ML）训练特征 | 一行一预测机会，键为实体和预测时刻 | 特征既要在业务时间内发生，也要在决策时可用。决策 10:00、交易 09:55、但 10:05 才入库，就不能用于这条样本。训练集同时记录输入和代码版本。 |
| 影像研究队列 | 一行一受试者、会话和模态的文件清单 | dbt 对清单、队列和质检结果做关系变换；图像配准和体素变换交给专用程序。训练和验证按受试者分组，重复扫描不能跨组。 |

在 Snowflake 或 Databricks 上，迁移时保留项目结构、粒度和业务测试，换 profile 和 adapter 后，重新核验 SQL 方言、类型、权限、运行成本、并发和增量策略。可以结合本站的[Snowflake 学习笔记]({{< relref path="/notes/snowflake-learning-guide" lang="zh" >}})和 [Databricks 支付实验]({{< relref path="/notes/databricks-guide" lang="zh" >}})对照：数据平台负责存、算和访问控制，dbt 项目负责维护一套可重跑的转换定义。

## 练习路线与外部资料 {#resources}

| 顺序 | 资料 | 读完后做一件事 |
|---|---|---|
| 1. 看一次完整流程 | {{< new-tab "dbt v2 官方入门" "https://docs.getdbt.com/docs/introduction?version=2" >}} · {{< new-tab "dbt Learn：Fundamentals（VS Code）" "https://learn.getdbt.com/courses/dbt-fundamentals-vs-code" >}} | 在自己的开发 schema 建一个模型，加一条测试，然后从血缘里找到它的输入。课程界面与本地命令可以不同。 |
| 2. 补 SQL 和建模 | {{< new-tab "DuckDB SQL 文档" "https://duckdb.org/docs/stable/sql/introduction" >}} · {{< new-tab "Kimball 四步维度设计" "https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/four-4-step-design-process/" >}} | 写下订单和支付各自的粒度。用 `row_number()` 找同键的当前版本，再用 `GROUP BY` 验证总额。 |
| 3. 阅读完整项目 | {{< new-tab "dbt Labs：Jaffle Shop" "https://github.com/dbt-labs/jaffle-shop" >}} · {{< new-tab "官方项目结构指南" "https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview" >}} | 挑一个 mart 逆着 `ref()` 追到源表；找出哪些步骤变了粒度。按示例仓库自身的版本说明运行。 |
| 4. 扩充质量检查 | {{< new-tab "dbt-utils 包与源码入口" "https://hub.getdbt.com/dbt-labs/dbt_utils/latest/" >}} · {{< new-tab "dbt 单元测试" "https://docs.getdbt.com/docs/build/unit-tests" >}} | 给日报写一条 `(ordered_on, channel)` 组合唯一测试，再为“部分支付”和“无支付”各补一个逻辑用例。 |
| 5. 串起调度与运行记录 | {{< new-tab "dbt 工作流指南" "https://docs.getdbt.com/best-practices/best-practice-workflows" >}} · {{< new-tab "Dagster 与 dbt 集成" "https://dagster.io/docs/integrations/libraries/dbt" >}} | 让造数、检查、构建按顺序运行，模拟输入过期和测试失败，确认作业何时停止、日报是否仍指向上一版。 |

想把它做成一个可展示的项目，可以加一道对账题：同时发布“按下单日归属的实收”和“按支付发生日归属的实收”，设计一个跨日付款，让两个结果有意不同。为两张表分别写明粒度、时间口径和可复现的预期值。最后换一个空目录跑一遍，使读者只拿到实验文件也能复现。
