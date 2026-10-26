# ccao2679.com · Hugo 版个人主页

首页采用左侧简介、右侧照片的布局，下面保留研究与项目时间流；手机端自动改为上下排列。使用系统字体和一个 CSS 文件，支持明暗模式。栏目页和话题页在构建时生成，部署在 GitHub Pages 上，不需要服务器。

## 目录结构

```
hugo.toml                 站点配置：名字、tagline、导航、GitHub 链接
preview.toml              只在生成离线预览时叠加（.html 平铺 + 相对链接 + CSS 内联）
content/
  about.md                科研与工程经历；availability 字段控制首屏下方的求职提示区（pinned: true）
  research/  projects/  coursework/  notes/
                          每个 .md 是一条条目，所在文件夹就是它的栏目
  publications/_index.md  论文页（内容来自 data/publications.toml）
  research/_index.md 等   栏目的一句话介绍（intro），显示在名字下面那一行
  contact/_index.md       联系方式页
layouts/                  模板（baseof / home / section / term / taxonomy / page + _partials）
layouts/_partials/home-hero.html  首页姓名、简短介绍、照片和 CV / 联系方式入口
static/images/chao-cao.jpg 首页个人照片，取景由 CSS 控制
assets/css/site.css       全部样式；颜色都在 :root 的 token 里（含 --fig-* 概览图配色）
assets/figs/<slug>.svg    条目的概览图，和 content 里的文件同名就会自动显示在条目左侧
assets/pubs/              论文主图
data/publications.toml    论文列表：first = true 的显示主图 + 讲解，其余简列
data/citations.json       引用数快照（构建时由 tools/citations.py 从 OpenAlex 刷新）
tools/figs/               概览图生成脚本（figures.py）
MAINTENANCE.md            每周维护的清单（定时任务按它执行）
.github/workflows/hugo.yml  push 到 master 后自动构建并发布
```

## 本地预览

```bash
# 安装 Hugo extended（macOS）
brew install hugo
# 或者：pip install hugo

hugo server          # 打开 http://localhost:1313 ，改文件会自动刷新
```

## 新增一个仓库条目

在 `content/projects/`（或 `research/`、`coursework/`）下新建 `repo-name.md`：

```yaml
---
title: "显示的标题"
date: 2026-10-01          # 只显示到月份；条目按日期倒序排列
repo: "https://github.com/HenryVarro666/repo-name"
link: "https://demo.example.com"   # 可选：有在线地址时标题链到这里
lang: Python              # 可选：右上角的语言
stars: 3                  # 可选：右上角的星数（0 不显示）
status: "updated monthly" # 可选：日期旁边的斜体状态
live: "Live at example.com."  # 可选：会有一个很慢的“呼吸”效果
topics: [speech]          # 点状下划线的话题标签（最多 2 个，从下表选），自动生成 /topics/xxx/ 页面
---
两三句话：做了什么、结论是什么。
```

### 话题（8 个大类）

按访客找东西的方向分，不按技术细节。新条目尽量从这里选，每条最多 2 个：

| 话题 | 放什么 |
|---|---|
| `brain-imaging` | 脑影像、皮层表面、MRI 相关的研究代码 |
| `speech` | TTS、ASR、语音翻译、说话人分离 |
| `llm` | 大模型应用、评测、提示工程、去标识化 |
| `ml-systems` | 推理引擎、部署、云上服务（vLLM、TensorRT-LLM、AWS） |
| `healthcare` | 医疗场景的应用 |
| `computer-vision` | 检测、跟踪等视觉任务 |
| `data-science` | 课程里的数据分析和预测模型 |
| `web` | 网站、Web 应用、本站 |

`research/`、`projects/`、`coursework/` 里的条目不会生成单独页面，标题直接链到 GitHub；`notes/` 里的文章会生成自己的页面。

## 常改的地方

| 想改什么 | 改哪里 |
|---|---|
| 名字下面那行字 | `hugo.toml` 里的 `params.tagline` |
| 导航栏目 | `hugo.toml` 里的 `[[menus.main]]` |
| 邮箱 / Google Scholar | 去掉 `hugo.toml` 里 `email`、`scholar` 两行前面的 `#` |
| 配色 | `hugo.toml` 里的 `palette`：`"next"` 是旧站 NexT 的灰色系（当前），删掉这行就换回青色默认；具体色值在 `assets/css/site.css` 顶部的 token 里 |
| 首屏简短介绍和照片 | `layouts/_partials/home-hero.html`、`static/images/chao-cao.jpg`；比例和取景在 `assets/css/site.css` 的首页样式里 |
| 经历和求职状态 | `content/about.md`：正文直接展示经历；`availability` 字段设置求职提示区的岗位、入职时间和地点 |
| 新增/修改论文 | `data/publications.toml`，主图放 `assets/pubs/` |
| 条目的概览图 | 改 `tools/figs/figures.py` 后运行它，输出到 `assets/figs/` |

## 部署到 www.ccao2679.com

域名在阿里云解析：`www` CNAME 到 `henryvarro666.github.io`，根域名 A 记录指向 GitHub Pages。DNS 不用改。

源码放在 `henryvarro666.github.io` 仓库的 **master** 分支（仓库默认分支）。旧 Hexo 生成的静态站留在 `hexo-archive` 分支。

1. 仓库 Settings → Pages → Build and deployment，把 Source 改成 **GitHub Actions**。Custom domain 保持 `www.ccao2679.com`，勾选 Enforce HTTPS。
   **先改这一步再推源码**：改 Source 不会下线当前网站，等新的构建发布后才替换。
2. 每次 push 到 `master`，Actions 会自动构建并发布，大约一分钟。也可以在 Actions 页手动点 Run workflow。

## 迁移旧博客文章（可选）

Hexo 文章的 front matter（`title`、`date`、`tags`）基本可以直接用：把 `.md` 放进 `content/notes/`，把 `tags:` 改名为 `topics:`，`categories:` 删掉即可。

## 生成离线预览

```bash
hugo --config hugo.toml,preview.toml -d preview
```

输出的是平铺的 `.html` 文件，链接都是相对路径，直接双击 `preview/index.html` 就能看。
