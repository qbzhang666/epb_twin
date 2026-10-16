# epb_twin 快速入门

epb_twin 把土压平衡盾构的逐环掘进记录转成有限元算例、推力扭矩分解和结果汇总，共六个经过测试的工具。可以接入语言模型
智能体，但模型只能看到窗口级的统计量，看不到逐环数据。新项目只需写四个文本文件，不需要写代码。

## 1. 安装
需要 Python 3.9 及以上，Windows、macOS、Linux 均可。
```
pip install -e .                                   # 基本功能：numpy、pandas、pyyaml
pip install -e ".[excel]"                          # 读取 .xlsx 记录
pip install -e ".[fe]" -c constraints-fe.txt       # 有限元推进模型：Kratos 10.4 与 gmsh 4（已测试版本）
```
集群上也可以用 `containers/epb_twin.def` 构建 Apptainer 镜像。

## 2. 先跑示范项目（约 5 分钟）
```
python -m epb_twin demo demo_project
python -m epb_twin check --project demo_project/project.yaml
python -m epb_twin bench --project demo_project/project.yaml --agent scripted   # 1.00，流程正常
python -m epb_twin bench --project demo_project/project.yaml --agent naive      # 分数低，说明数据偏差确有影响
```
示范项目的掘进、地层和数据全部为随机生成的虚构数据。

## 3. 配置自己的项目
```
my_project/
  project.yaml      # 文件位置、运行方式（Slurm 或本机）、数据边界
  adapter.yaml      # 记录的列名、单位、地层代码、掘进模式词表
  machine.yaml      # 盾构参数
  ground.yaml       # 地层材料、默认分层、荷载模型的地层类别
  data/record.csv   # 掘进记录（只留在本机）
```
- **列名和单位**：在 `adapter.yaml` 的 `columns` 写"标准名: 你的列名"，`units` 写与标准单位不同的字段，例如
  `face_crown_bar: kPa`、`thrust_kn: MN`、`cutterhead_torque_mnm: kN m`。中文表头的完整示例见
  `epb_twin/adapters/chinese_headers.yaml`，包括"土压平衡/半土压/气压"等模式名和"填土/黏土/强风化/中风化"等地层词。
- **检查映射**：`python -m epb_twin check --project project.yaml`，只输出计数，不输出逐环数值。
- 标准字段、单位和含义见 `epb_twin/schema.py`；各配置文件的完整说明见 `docs/USER_GUIDE.md`。

## 4. 使用
```
python -m epb_twin call --project project.yaml list_windows '{"sort_by": "soil_thick_mean", "top": 5}'
python -m epb_twin call --project project.yaml build_window '{"drive": "D1", "ring_from": 120, "name": "W1"}'
```
`write_job` 生成算例表和运行脚本（Slurm 阵列或本机逐个运行）。先运行输出里的 `mesh_commands` 生成网格，再提交作业。

## 5. 语言模型（可选）
- 本地开源模型（Ollama）：`ollama pull qwen3.6:27b` 后运行
  `python -m epb_twin bench --project P --agent openai --model qwen3.6:27b`
- 其他兼容 OpenAI 接口的服务：加 `--base-url http://主机:端口/v1`
- Claude API：`--agent anthropic --model 模型名`
- 支持 MCP 的客户端：服务命令为 `python -m epb_twin mcp --project /路径/project.yaml`

所有工具调用都记录在 `_audit/tool_calls.jsonl`，便于数据所有方核查。

## 6. 许可证与引用
BSD 3-Clause。第三方依赖的许可证见 THIRD_PARTY.md（gmsh 为 GPL，仅在生成网格时调用）。引用格式见 CITATION.cff。
