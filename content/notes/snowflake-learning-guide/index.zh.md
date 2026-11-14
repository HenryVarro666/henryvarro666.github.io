---
title: "Snowflake 学习笔记：从第一条 SQL 到订单分析管道"
date: 2026-11-13T18:08:00-06:00
description: "给第一次接触云数据仓库的人。用一套合成订单练习建仓、装载文件、清洗去重和增量更新，再用 Query Profile 判断该优化 SQL 还是调整算力。附完整脚本、图解和官方学习路线。"
topics: [data-science, ml-systems]
---

电商系统发来九行订单记录。其间有一次重发、一条迟到的旧记录、一笔退款，还有一个金额是 `oops`。直接做 `SUM(amount)` 能不能算出当天有多少已付款订单、总额是多少？这份笔记从这个问题出发。你会建一个小型 Snowflake 数据仓库，把原始记录变成可以核对、可以重跑的日报，然后学会判断慢查询到底慢在哪里。

**怎么读。** 会写 `SELECT` 就能开始；没学过窗口函数也可以跟着例子走。第一次花约 20 分钟读架构、跑通第一条查询；再留 45–60 分钟完成订单实验。订单实验使用合成数据。[下载 SQL 练习脚本](/notes/snowflake-learning-guide/orders-lab.sql) · [下载九行订单数据（CSV，逗号分隔值）](/notes/snowflake-learning-guide/orders.csv)。

## 1. 先理解 Snowflake 在做什么 {#mental-model}

Snowflake 是托管在公有云上的数据平台。这里先把它当作分析型数据库：订单应用通常用事务数据库记录付款和状态变更，Snowflake 把多个来源的历史数据放在一起，回答诸如「上个月哪个渠道带来的已付款订单最多？」的问题。前者典型是在线事务处理（Online Transaction Processing，OLTP），后者典型是在线分析处理（Online Analytical Processing，OLAP）。本文使用标准 Snowflake 表；Hybrid Tables 是针对低延迟事务读写设计的另一种表。

![登录和授权由云服务层负责。报表与加工分别使用独立的计算仓库，读取同一份持久化订单表。](architecture.svg "图 1｜两个计算仓库可以读同一张表。暂停报表仓库，订单仍在存储层；恢复仓库后可以继续查询。")

| 层 | 具体做什么 | 在练习里怎么认 |
|---|---|---|
| 云服务（Cloud Services） | 身份验证、权限检查、管理元数据、解析和优化查询 | 角色能否查表、查询怎样规划 |
| 计算（Compute） | Virtual Warehouse 提供计算资源，执行查询、装载和加工 | `SF_LEARN_ORDERS_WH` 决定本次用哪份算力 |
| 存储（Storage） | 标准表以压缩列式数据持久化，Snowflake 自动划分微分区 | `SF_LEARN_ORDERS.CORE.ORDERS` 是数据对象，仓库暂停也不会消失 |

**最容易混淆的两个词：** Database 是对象的命名空间，Warehouse 是算力。把同一张表交给另一个 warehouse 查询，改变的是资源和成本归属。两个 warehouse 可以隔开报表与加工的计算负载，但要分别管理权限和使用量。[Snowflake 架构与对象](https://docs.snowflake.com/en/user-guide/intro-key-concepts)

常用对象按这个顺序认：

| 对象 | 记法 |
|---|---|
| Account（账户） | 登录、账户级角色和计费所在的边界 |
| Database → Schema → Table / View | `数据库.模式.对象`，例如 `SF_LEARN_ORDERS.MART.DAILY_PAID_SALES`。普通视图保存查询定义 |
| Role（角色） | 权限的集合。用户切换角色，相同 SQL 可能得到权限错误 |
| Stage（暂存位置） / File Format（文件格式） | 前者告诉 Snowflake 文件在哪，后者告诉它怎样解析 CSV、JSON（JavaScript Object Notation）或 Parquet |

语句里不带双引号的标识符通常按大写解析；带双引号则要保留原来的大小写。教程统一用不加引号的名字。SQL 字符串用单引号。

## 2. 第一次运行：角色、仓库和查询 {#first-query}

在个人试用账户或获准的练习环境动手。按[试用账户说明](https://docs.snowflake.com/en/user-guide/admin-trial-account)，试用以注册后 30 天或额度耗尽两者先到为止；建仓查询前先看账户剩余额度。打开 Snowsight，在 **Projects → Workspaces → My Workspace** 新建 SQL File，检查编辑器里选中的角色和仓库。旧教程里常写 Worksheets，现在用 [Workspaces](https://docs.snowflake.com/en/user-guide/ui-snowsight/workspaces#run-your-first-sql-query) 即可。

### 建一个专用沙盒

下面用个人沙盒里的 `SYSADMIN` 建临时练习环境。在公司账户，让管理员提供已有的练习库、warehouse 和角色，再替换相应的名字；普通使用者通常不需要账户管理员权限。在同一个 SQL 会话里按小节执行。若任何 `CREATE` 发现同名对象已存在，先停下检查，换一组名字或确认是在继续自己的实验。

```sql
USE ROLE SYSADMIN;

CREATE WAREHOUSE sf_learn_orders_wh
  WAREHOUSE_TYPE = 'STANDARD'
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  STATEMENT_TIMEOUT_IN_SECONDS = 120;

CREATE DATABASE sf_learn_orders DATA_RETENTION_TIME_IN_DAYS = 1;
USE WAREHOUSE sf_learn_orders_wh;
USE DATABASE sf_learn_orders;
CREATE SCHEMA raw;
CREATE SCHEMA core;
CREATE SCHEMA mart;

SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE();
```

`AUTO_SUSPEND = 60` 的单位是秒，空闲时暂停；需要计算的下一条语句可以自动恢复。超时用于限制单条语句的运行时间，不能代替预算。查询报「对象不存在或无权限」时，先打印当前角色、数据库和 schema，试用全限定名；如果名字正确，再查角色是否同时拥有仓库、上层对象和目标对象所需的权限。

### 先查一张已有的表

Snowflake 提供行业基准样例数据。如果账户里已安装 `SNOWFLAKE_SAMPLE_DATA` 且角色有访问权限，可以先数一数「每种订单优先级各有多少笔？」：

```sql
SELECT o_orderpriority AS priority, COUNT(*) AS orders
FROM snowflake_sample_data.tpch_sf1.orders
GROUP BY o_orderpriority
ORDER BY orders DESC, priority
LIMIT 5;
```

`TPCH_SF1` 是事务处理性能委员会的 TPC-H 基准样例，`SF1` 表示 Scale Factor 1。先用小规模数据；`LIMIT 5` 只限制返回行数，这条查询仍要先做聚合。没有样例库时可以直接做下一节，实验不依赖它。[样例数据与访问方式](https://docs.snowflake.com/en/user-guide/sample-data)

## 3. 主实验：从原始订单到每日实收 {#orders-lab}

日报的**口径和粒度**先写在纸上。订单表每个 `order_id` 一行，保存当前已接受的最新版本。日报按原下单日期统计当前状态为 `PAID` 的订单笔数与金额，单位为美元（统一币种）。退款后的订单从对应下单日的已付款指标里移除。因此这是**按订单日回看当前状态**的报表，不能用它做财务对账；如果要追踪每天实际收退款，应该另建按支付或退款交易时间记账的流水事实表。

![九行 CSV 或 INSERT 进入原始表。校验失败的记录可从原始表查到；有效行经过去重和 MERGE，进入订单表，再由视图计算日报。](pipeline.svg "图 2｜RAW 保留原值；CORE 以订单为粒度；MART 以日期为粒度。重复处理一批数据时，对照 CORE 和 MART 的结果判断有没有重复记账。")

### 步骤 A：把数据放进 RAW

**第一次建议直接用 SQL。** 打开下载的脚本，运行第 1 节，然后在第 2 节建 `raw.order_events` 并执行九行 `INSERT`。金额、日期和源版本先按文本保存；上游出现坏值时有原始证据可查。

**想练 CSV 装载时，用下列流程替代九行 `INSERT`。** 先建好同一张空原始表，再建 named internal stage（命名内部暂存区）和 CSV 格式。在 Snowsight 用 **Ingestion → Add Data → Load files into a Stage**，选择 `SF_LEARN_ORDERS.RAW.ORDERS_INBOX`，把下载的 `orders.csv` 上传到 stage 根路径。浏览器里的 SQL 编辑器不能直接用 `PUT` 读取本机路径；命令行用户可用 Snowflake 命令行工具（CLI）的 `snow sql` 执行 `PUT`。[Snowsight 上传文件](https://docs.snowflake.com/en/user-guide/data-load-local-file-system-stage-ui) · [PUT 文档](https://docs.snowflake.com/en/sql-reference/sql/put)

```sql
CREATE STAGE raw.orders_inbox;
CREATE FILE FORMAT raw.orders_csv
  TYPE = CSV
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"';

-- Upload orders.csv to the named stage before running the following.
LIST @raw.orders_inbox;

COPY INTO raw.order_events
  FROM @raw.orders_inbox
  FILES = ('orders.csv')
  FILE_FORMAT = (FORMAT_NAME = 'SF_LEARN_ORDERS.RAW.ORDERS_CSV')
  VALIDATION_MODE = RETURN_ALL_ERRORS;

COPY INTO raw.order_events
  FROM @raw.orders_inbox
  FILES = ('orders.csv')
  FILE_FORMAT = (FORMAT_NAME = 'SF_LEARN_ORDERS.RAW.ORDERS_CSV')
  ON_ERROR = 'ABORT_STATEMENT';

SELECT COUNT(*) AS raw_rows FROM raw.order_events;
```

预校验应无报错，正式装载应显示 9 行，表中也应是 9 行。预校验只检查文件能否按指定格式和目标类型装载。`oops` 可以装进文本列，所以还需要下面的业务校验。Snowflake 会利用装载元数据减少同一文件重复装载；同一个订单也可能出现在不同文件里，仍要做业务键去重。不要为了重试默认加 `FORCE = TRUE`。[COPY INTO 表的语法和装载选项](https://docs.snowflake.com/en/sql-reference/sql/copy-into-table)

### 步骤 B：校验、去重，再做 MERGE

继续运行脚本第 2 节的视图和建表，先查 `raw.order_events_checked` 的 `reject_reason`。它用 `TRY_TO_DATE` 和 `TRY_TO_DECIMAL` 把转换失败变成 `NULL`，并用格式规则拦下多于两位小数或非整数字符串版本。样例只允许非负金额和正整数源版本。预期恰好一行坏数据：`ingest_id = 8`、`O1005`、`BAD_AMOUNT`。生产中应记录失败比例，修正后重新投递，并为相同版本但不同内容的记录设告警。

先看三组容易出错的记录：

| 订单 | RAW 中发生了什么 | CORE 要保留什么 |
|---|---|---|
| `O1001` | 版本 2（120.00）先到，版本 1（100.00）后到 | 版本 2。到达更晚不代表上游版本更新 |
| `O1003` | 先付款，后退款 | 版本 2、`REFUNDED`。保留订单，日报不计入 |
| `O1004` | 同版本、同内容发了两次 | 只留一行 |

下面是脚本第 3 节的关键语句。先在当前批次里按 `order_id` 分组，版本最大的一条胜出；同版本同内容的重发用唯一的 `ingest_id` 打破平局。再和目标表比较版本：旧版本不能把已接受的新版本覆盖掉。

```sql
SET batch_first = 1;
SET batch_last = 9;

MERGE INTO core.orders AS t
USING (
  SELECT order_id, order_date, status, amount, source_version
  FROM raw.order_events_checked
  WHERE reject_reason IS NULL AND ingest_id BETWEEN $batch_first AND $batch_last
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY order_id ORDER BY source_version DESC, ingest_id DESC
  ) = 1
) AS s ON t.order_id = s.order_id
WHEN MATCHED AND s.source_version > t.source_version THEN UPDATE SET
  t.order_date = s.order_date,
  t.status = s.status,
  t.amount = s.amount,
  t.source_version = s.source_version
WHEN NOT MATCHED THEN INSERT (order_id, order_date, status, amount, source_version)
  VALUES (s.order_id, s.order_date, s.status, s.amount, s.source_version);
```

`WHERE` 先筛掉不属于本批或校验失败的行，`QUALIFY` 在窗口函数计算后筛选。`MERGE` 不替你确定哪个源记录应该胜出；同一目标键匹配多条源记录可能使更新报错，多条未匹配源记录也可能全部插进去。标准表上的主键、唯一键和外键不强制执行，所以还要查询目标表有无重复键。如果编辑器重新连接，先重新设置两个批次变量，再执行 `MERGE`。[QUALIFY](https://docs.snowflake.com/en/sql-reference/constructs/qualify) · [MERGE](https://docs.snowflake.com/en/sql-reference/sql/merge) · [标准表和 Hybrid Tables 的约束](https://docs.snowflake.com/en/sql-reference/constraints-overview)

这里的策略是保留最后一个**有效**版本。如果新版无效，旧的有效状态可能继续显示，直到修复并补发。实际系统应明确这种情况下是保留旧状态并告警、暂停更新，还是暂时下线该订单；不要让分析人员把旧值误当成最新值。脚本假设同键同版本的有效事件内容相同，且只有一个写入者。

### 步骤 C：对账，再用迟到事件测试重跑

`mart.daily_paid_sales` 是普通视图。从 `core.orders` 取 `PAID`，并按 `order_date` 计算 `COUNT(*)` 和 `SUM(amount)`。你可以这样检查：

```sql
SELECT COUNT(*) AS order_rows FROM core.orders;

SELECT order_id, COUNT(*) AS n
FROM core.orders
GROUP BY order_id
HAVING COUNT(*) > 1;

SELECT * FROM mart.daily_paid_sales ORDER BY order_date;
```

| 日期 | 批次 A：笔数 / 金额（美元） | 再处理批次 B：笔数 / 金额（美元） |
|---|---:|---:|
| 2026-11-01 | 2 / 195.50 | 2 / 190.50 |
| 2026-11-02 | 1 / 80.00 | 2 / 105.00 |

批次 A 后，CORE 应有 5 行，重复键查询返回 0 行。第一天是 `120.00 + 75.50 = 195.50`；退款后的 `O1003` 不参与。第二天只能数一次 `O1004`。

接着运行脚本第 4 节：**只插入一次**批次 B，再重跑上面的 `MERGE` 两次。第一次应更新 `O1002` 的金额为 70.50，插入 `O1007`（25.00），忽略 `O1001` 迟到的旧版本。第二次应没有新增或更新，日报保持右栏结果。这叫**幂等**：重复处理同一输入，业务结果不再变化。这里验证的是加工步骤的幂等性；生产还需用提交成功后才推进的检查点、稳定的源事件标识和可审计的批次记录来管理重试。

## 4. 常用 SQL：处理数组、保留时间语义 {#useful-sql}

### JSON：一行订单可以展开成多行商品

`VARIANT` 是 Snowflake 存半结构化数据的类型。用 `PARSE_JSON` 把字符串转为结构，再用冒号取键、显式转换类型。定位到数组后，`LATERAL FLATTEN` 会按左侧每条记录展开。订单一行变成商品一行，粒度变了。

```sql
WITH orders AS (
  SELECT PARSE_JSON('{
    "order_id": "O1001",
    "items": [{"sku":"A","qty":2,"price":40.00}, {"sku":"B","qty":1,"price":40.00}]
  }') AS payload
)
SELECT o.payload:order_id::VARCHAR AS order_id,
       f.value:sku::VARCHAR AS sku,
       f.value:qty::NUMBER(10, 0) * f.value:price::NUMBER(12, 2) AS line_amount
FROM orders AS o, LATERAL FLATTEN(INPUT => o.payload:items) AS f
ORDER BY sku;
```

两行结果是 A = 80.00、B = 40.00。展开后如果对原订单总额求和，120.00 可能会被算成 240.00。用行项目金额聚合，或先回到订单粒度再与订单表关联。空数组默认不会产生行；需要保留左侧记录时，研究 `FLATTEN` 的 `OUTER => TRUE`，同时写下空数组、字段缺失和 JSON null 的处理规则。高频筛选和关联字段通常提取成定型列更容易控制语义。[查询半结构化数据](https://docs.snowflake.com/en/user-guide/querying-semistructured) · [FLATTEN](https://docs.snowflake.com/en/sql-reference/functions/flatten)

### 日期：报表中的「当天」要先定时区

订单练习的日期已由上游提供，所以用 `DATE`。有真实事件流时，保留原始事件时间和入库时间，再按明确的业务时区推导报表日期。例如不含时区的 `TIMESTAMP_NTZ` 已约定用协调世界时（UTC）存储，下面这条语句应返回 `2026-11-01`：

```sql
SELECT CONVERT_TIMEZONE(
  'UTC', 'America/Chicago', '2026-11-02 02:00:00'::TIMESTAMP_NTZ
)::DATE AS reporting_date;
```

不同地区的午夜和夏令时变化用名称指定，例如 `America/Chicago`。训练集还需要区分用户当时发生了什么、系统当时已经看到了什么。[CONVERT_TIMEZONE](https://docs.snowflake.com/en/sql-reference/functions/convert_timezone)

## 5. 查询慢了：先定位，再调算力 {#performance-cost}

![日期筛选只需读取范围可能命中的微分区；区间完全在过滤范围之外的分区可以跳过。](pruning.svg "图 3｜筛选 11 月 2 日。读取日期范围有交集的微分区，再按行筛选。图中的四个分区用于说明原理，实验中可以用 Query Profile 核对实际扫描量。")

Snowflake 自动把标准表切成微分区（micro-partitions），记录各分区的列范围等元数据。查询含 `WHERE order_date >= '2026-11-02'::DATE AND order_date < '2026-11-03'::DATE` 时，引擎有机会排除日期范围不相交的分区。若同一天的记录散落在许多分区，裁剪（pruning）可能很少。九行订单太小，用于理解口径；想看差别，请选有足够微分区的样例表。[微分区与裁剪](https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions)

### 在 Query Profile 里看证据

在 Snowsight 的 Query History 找到语句，打开 Query Profile，先记下查询 ID 和 warehouse。分清**等待资源、等待编译和实际执行**，然后找扫描、Join、聚合里耗时最大的节点。

| 观察到什么 | 先做什么 |
|---|---|
| 排队时间长，但开始执行后很快 | 检查是否和批处理共享 warehouse；按负载拆分，或评估多集群仓库。多集群主要解决并发，有版本要求 |
| 扫了大部分分区，实际只需要少量数据 | 核对筛选范围和扫描列；查看 `Partitions scanned / total` 与 `Bytes scanned`。大表长期出现这个模式再评估聚簇键及维护费用 |
| Join 后输出行数陡增 | 核对业务粒度、Join 键唯一性和漏掉的连接条件。加大计算资源不会修正算错的订单笔数 |
| 排序或聚合大量 spill 到远端存储 | 先缩小输入、去掉不必要的列或降低中间结果规模；问题仍在，再比较更大 warehouse |
| 第二次运行突然快很多 | 区分已保存的查询结果与 warehouse 本地数据缓存。比较算力时，在测试会话用 `ALTER SESSION SET USE_CACHED_RESULT = FALSE;`，并记录仓库冷暖状态；这个开关只绕过结果复用 |

[Query History / Query Profile](https://docs.snowflake.com/en/user-guide/ui-snowsight-activity) · [Warehouse 选择](https://docs.snowflake.com/en/user-guide/warehouses-considerations) · [已保存的结果](https://docs.snowflake.com/en/user-guide/querying-persisted-results)

**做一次可以比较的实验。** 如果可访问样例库，用下面的范围聚合查 1995 年 1 月订单。再跑一遍去掉 `WHERE` 的版本，两次分别在 Profile 的 TableScan 中记录扫过的分区数和字节数。查询回答的问题不同，这里比较的是扫描范围。若想衡量某次改写是否更快，两条语句还要先验证结果一致。

```sql
ALTER SESSION SET USE_CACHED_RESULT = FALSE;

SELECT SUM(o_totalprice) AS total_price
FROM snowflake_sample_data.tpch_sf1.orders
WHERE o_orderdate >= DATE '1995-01-01'
  AND o_orderdate < DATE '1995-02-01';

-- After measuring both queries, restore the default behavior.
ALTER SESSION UNSET USE_CACHED_RESULT;
```

### 费用：看使用量，别只看一次延迟

运行中的虚拟仓库按秒计费，每次启动或恢复至少计一分钟。举例：**假设**单集群速率为 1 credit/小时，连续运行 12 分钟约 0.20 credit；若是 12 次各 10 秒、每次都从暂停状态启动，就会分别触发至少一分钟，合计也至少 0.20 credit。例子没有计入其他服务，也不是账户报价。长时间空闲时及时暂停可以节省仓库计算费；暂停后存储费用仍可能发生，serverless 功能也可独立计费。[计算费用](https://docs.snowflake.com/en/user-guide/cost-understanding-compute)

实际对比时，在相同角色、数据范围和缓存条件下多跑几次，记录延迟和扫描量，并核对 warehouse 的用量。资源监视器（Resource Monitor）可限制指定用户管理的仓库用量；包括 Snowpipe 在内的 serverless 功能得另外看用量和预算，不能靠暂停练习仓库停止。[资源监视器的覆盖范围](https://docs.snowflake.com/en/user-guide/resource-monitors)

## 6. 走出实验：管道、权限和恢复 {#production}

### 什么时候换工具

从需要处理的输入和故障出发。一天一批文件，一次 `COPY` 和定期加工就能说明业务，先保证可查、可重跑。再按需要选择：

| 需求 | 先考虑 | 要核对的边界 |
|---|---|---|
| 定时装 CSV / Parquet，能按文件重跑 | Stage + `COPY INTO` | 文件名与批次唯一性、坏行隔离、装载历史 |
| 对象存储不断到新文件 | Snowpipe | 文件级持续装载；查看延迟、文件大小、重复事件与 serverless 费用 |
| 源程序持续发送单行事件 | Snowpipe Streaming | 无需先落文件；定义渠道、源偏移量和重放策略 |
| 用一条 `SELECT` 定义经常查询的派生表 | Dynamic Table（动态表） | `TARGET_LAG` 是期望新鲜度，不是固定调度或承诺；检查刷新模式、支持的 SQL、实际滞后和费用 |
| 需要按变更写入目标，做定制 upsert、删除或多步处理 | Stream + Task | Stream 跟踪数据变更偏移，不是永久的消息队列；读 `SELECT` 不推进偏移，成功提交的消费型数据变更才推进。独立消费者通常用独立 stream |
| 多人维护指标，需要版本控制、测试、文档和依赖图 | dbt + Snowflake | dbt 负责建模与测试，Snowflake 执行 SQL。为 `core.orders` 写 `unique`、`not_null`，为状态写允许值测试 |

[装载方案](https://docs.snowflake.com/en/user-guide/data-load-overview) · [Dynamic Tables 决策指南](https://docs.snowflake.com/en/user-guide/dynamic-tables/decision-guide) · [Streams](https://docs.snowflake.com/en/user-guide/streams-intro) · [dbt 的 Snowflake 配置](https://docs.getdbt.com/docs/local/connect-data-platform/snowflake-setup)

### 报表只读，管道单独写

用基于角色的访问控制（Role-Based Access Control，RBAC）拆开不同职责。分析角色只需要 warehouse 的 `USAGE`，库和 schema 的 `USAGE`，以及已授权视图的 `SELECT`；应由能授权的管理员授予这些权限。报表使用者不需要直接读 `RAW`。管道角色才获得写入权，敏感列遮盖和按行授权按需要加。上线前切换成真实报表角色验一遍「能看应看的视图，读不到原始表」；账户管理员跑通不能替代这一步。[访问控制](https://docs.snowflake.com/en/user-guide/security-access-control-overview)

### 误操作时先看历史

Time Travel 可以在对象的保留期内查询过去的数据；zero-copy clone 起初共享已有存储，后续修改可能产生额外存储。练习时，把误改语句自己的 Query ID 填到下面的占位符，先查旧状态，确认受影响的行，再决定怎么恢复。

```sql
SELECT order_id, status, amount
FROM sf_learn_orders.core.orders
BEFORE (STATEMENT => '<query_id_of_the_change>')
ORDER BY order_id;
```

`BEFORE` 取指定变更之前的版本。保留期受表类型、账户版本和配置影响。Time Travel 是在可用保留期内的恢复工具；长期审计或外部恢复需要独立方案。[Time Travel](https://docs.snowflake.com/en/user-guide/data-time-travel) · [AT / BEFORE](https://docs.snowflake.com/en/sql-reference/constructs/at-before)

## 7. 应用迁移：BI、数据科学和高频服务 {#applications}

| 场景 | 从哪个对象开始 | 首要验证 |
|---|---|---|
| BI（Business Intelligence，商业分析） | 读 `MART.DAILY_PAID_SALES` 的只读仪表盘，统一币种与时区 | 在图上任选一天，用原始订单回算；找出退款会改变哪一天 |
| 数据科学 | 用 Snowpark Python 在库内筛选、关联和聚合，再取小结果 | 训练样本只使用标签时点之前且当时已可见的数据。当前订单状态会泄漏之后才发生的退款 |
| 实时风控 | 在仓库做历史特征、离线评估、漂移监测 | 如果在线请求有很紧的尾延迟目标，单独评估在线特征服务或低延迟存储；不要用分析 SQL 的平均延迟当线上承诺 |

想在本地 Python 里接着用练习结果，先在虚拟环境里安装 `snowflake-snowpark-python`，再配置名为 `orders_lab` 的 Snowflake 连接（角色能读取 MART，连接选用获准的 warehouse）。请按规范配置认证方式，代码里只放连接名。[安装 Snowpark Python](https://docs.snowflake.com/en/developer-guide/snowpark/python/setup)

```python
from snowflake.snowpark import Session

session = Session.builder.config("connection_name", "orders_lab").create()
try:
    rows = (
        session.table("SF_LEARN_ORDERS.MART.DAILY_PAID_SALES")
        .sort("ORDER_DATE")
        .limit(10)
        .collect()
    )
    for row in rows:
        print(row)
finally:
    session.close()
```

Snowpark DataFrame 的转换通常延迟到动作（例如 `collect()`）才执行。先在 Snowflake 里限制范围；`collect()` 会把结果带回 Python，适合这张小日报，不适合把全量事件拉到笔记本内存。[创建 Snowpark 会话](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session) · [Snowpark DataFrame](https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes)

## 8. 自测与收尾 {#practice}

1. **改一笔订单。** 把 `O1007` 的下一版本改成退款，再同时送一个更旧的付款版本。预测哪一天的笔数和金额会变化，运行一次后重跑，核对日报总额。
2. **练一次关联。** 新建一张订单行项目表，给 `O1001` 加两件商品。直接 Join 后对订单金额求和，解释为什么错，再写出按订单粒度求和的版本。
3. **练一次诊断。** 在样例表比较带日期范围与不带范围的查询。记录各自扫描的分区、字节数和耗时，说明是否值得调整 warehouse。只比较一次运行时间不算完成。
4. **练一次交接。** 写下日报口径、批次边界、拒绝数据的位置和用只读角色重现结果的步骤。交给另一个人，看他能否独立复算。

练完先暂停 `SF_LEARN_ORDERS_WH`；若已自动暂停，跳过第一条。若确认它和数据库只用于这次实验，可以删除。在自己的沙盒中核对对象再逐条执行。

```sql
ALTER WAREHOUSE sf_learn_orders_wh SUSPEND;
-- Only after checking these are your lab's disposable objects:
DROP DATABASE sf_learn_orders;
DROP WAREHOUSE sf_learn_orders_wh;
```

## 9. 值得按顺序读的资料 {#further-reading}

| 顺序 | 资料 | 做完应该会什么 |
|---|---|---|
| 1 | [Snowflake in 20 minutes](https://docs.snowflake.com/en/user-guide/tutorials/snowflake-in-20minutes) | 用自己的角色和 warehouse 建库、装载、查询。遇到旧界面时对照 [Workspaces 指南](https://docs.snowflake.com/en/user-guide/ui-snowsight/workspaces-working) |
| 2 | [Snowflake University：Data Warehousing Workshop](https://learn.snowflake.com/en/courses/OD-ESS-DWW/) | 从输入文件走到表；做 CSV 和 JSON 实验，不只看演示 |
| 3 | [官方动手教程合集](https://docs.snowflake.com/en/learn-tutorials) | 选一个真实来源，例如 Amazon S3 或 Azure 对象存储，练一次从暂存区装载 |
| 4 | [Query Profile 与查询历史](https://docs.snowflake.com/en/user-guide/ui-snowsight-activity) | 从算子和扫描指标解释一次慢查询，用数据说明改动是否有效 |
| 5 | [Snowflake University：Hands-On Essentials](https://learn.snowflake.com/en/pages/hands-on-essentials-track/) | 做完 Badge 1 后按方向挑 Data Engineering、Data Science 或 Data Applications 的下一门 |
| 6 | [《The Snowflake Elastic Data Warehouse》原论文，SIGMOD 2016](https://pages.cs.wisc.edu/~remzi/Classes/739/Papers/p215-dageville-snowflake.pdf) | 理解共享存储、多集群计算背后的取舍。论文用于读设计动机，当前功能以产品文档为准 |

如果只选一个后续作业：用另一批合成订单重做日报，让同伴从 CSV 独立算出同一组数字，再说明你的管道遇到一批重复文件时会发生什么。
