# 载波相位整周序列唯一性判定服务（carrier-phase-unwrapper）

授时站的载波相位观测只保留模 `M` 的余数 `r_i`。若直接逐点展开（unwrap），
周跳会让时间线漂移出整数个 `M`，得到错误的时间线。本服务在已知起点
`x_0`（满足 `x_0 ≡ r_0 (mod M)`）与逐点增量闭区间 `[lo_i, hi_i]` 的约束下，
重建最优整数时间线，并判定最优解是否**唯一**：

- 唯一 → 返回完整时间线 `x`、总代价与增量序列；
- 不唯一 → 返回 `ambiguous` 及按 `x` 字典序最小的两条最优见证；
- 无解 → 返回 `impossible` 及首个无候选的时刻。

纯后端实现：Python 3.13 + FastAPI + Pydantic，全程**精确整数运算，禁用浮点**，
允许负的 `x` 与负增量，闭区间端点一律包含。

## 1. 数学模型

给定整数 `M`（`2 ≤ M ≤ 10^9`）、历元数 `n`（`1 ≤ n ≤ 20000`）、余数序列
`r_0..r_{n-1}`（`0 ≤ r_i < M`）、起点 `x_0`（`x_0 ≡ r_0 (mod M)`，可为负），
以及 `n-1` 个增量闭区间 `[lo_i, hi_i]`（`i = 1..n-1`，`lo_i ≤ hi_i`，端点包含）。

求整数序列 `x_0..x_{n-1}` 满足：

1. `x_i ≡ r_i (mod M)`；
2. 增量 `d_i = x_i − x_{i−1}` 满足 `lo_i ≤ d_i ≤ hi_i`；
3. 最小化相邻增量变化量绝对值之和

   ```
   cost = Σ_{i=2}^{n-1} |d_i − d_{i−1}|        （n < 3 时为空和，恒为 0）
   ```

由约束 1 可知可行增量恰为满足 `d ≡ r_i − r_{i−1} (mod M)` 且落在
`[lo_i, hi_i]` 内的整数；输入契约要求每个区间**至多 64 个**这样的模差候选，
超过即属非法输入（见 §5 错误码）。

## 2. 算法

设 `C_i` 为时刻 `i` 的候选增量集合（按升序，公差为 `M` 的等差数列，`|C_i| ≤ 64`）。

- **候选生成**：最小候选 `d₀ = lo_i + ((r_i − r_{i−1} − lo_i) mod M)`，
  之后步长 `M` 直至 `hi_i`；个数 `= ⌊(hi_i − d₀)/M⌋ + 1`（`d₀ > hi_i` 时为 0）。
  全程 Python 整数取模，负数语义正确。
- **逆向动态规划**：`g_i(c) = min_{c′∈C_{i+1}} |c′ − c| + g_{i+1}(c′)`，
  `g_{n-1}(·) = 0`。这正是把第 `i+1` 行的值做一维 **L1 距离变换**后在
  `C_i` 上求值。用左右两遍单调扫描（`≤ c` 的源贡献 `c + min(h_j − c′_j)`，
  `> c` 的源贡献 `−c + min(h_j + c′_j)`）在 `O(|C_i| + |C_{i+1}|)` 内完成一步，
  全程 `O(n·K)`（`K ≤ 64`），规模上限 `n = 20000, K = 64` 时约 2.6×10⁶ 次
  整数运算，实测约 0.5 秒。
- **计数截断**：与值同步传播最优延续方案数，超过 2 即截断为 2
  （`count = min(2, …)`）。左右两个源集合不相交，不会重复计数。
- **见证重构**：利用保存的 `g` 行从 `i = 1` 起贪心——每步取仍能满足最优
  子结构方程 `|c − d_{i−1}| + g_i(c) = g_{i−1}(d_{i−1})` 的最小候选，得到
  字典序最小序列；第二条约从后往前找**最晚可偏离位置**，取该处满足同一
  方程的最小更大候选，再贪心补全。`x_0` 固定时，`x` 的字典序与增量序列
  的字典序一致，故两条见证即按 `x` 字典序最小的两条。
- 计数为 1 → `unique`；计数 ≥ 2（截断后恰为 2）→ `ambiguous`；
  某时刻候选为空 → `impossible`。

## 3. 复算规则（验收口径）

给定响应后，可按以下规则独立复算，全部只用整数：

1. **候选**：时刻 `i`（`1 ≤ i ≤ n−1`，对应数组 `lo[i-1]/hi[i-1]`）的候选为
   `{ d ∈ [lo_i, hi_i] : d ≡ (r_i − r_{i−1}) mod M }`；最小者
   `d₀ = lo_i + ((r_i − r_{i−1} − lo_i) mod M)`，个数 `⌊(hi_i − d₀)/M⌋ + 1`。
   个数为 0 的最小时刻即 `first_empty_index`；个数 > 64 属非法输入。
2. **合法性**：每条返回的 `x` 须满足 `x_0` 等于请求起点、`x_i mod M == r_i`
   （Python 取模语义，负数 `x` 的余数落在 `[0, M)`）、
   `x_i − x_{i−1} == increments[i−1] ∈ [lo_i, hi_i]`。
3. **代价**：`cost = Σ_{i=2}^{n-1} |d_i − d_{i−1}|`，`n < 3` 时为 0；
   可用 §2 的 `g` 递推（或 `O(n·K²)` 直接 DP）独立重算并比对。
4. **计数**：最优时间线总数截断为 2——恰 1 条为 `unique`，≥ 2 条为
   `ambiguous`（响应只携带字典序最小的两条见证，不返回总数）。
5. **见证次序**：`ambiguous` 的两条见证按 `x` 字典序升序，且是全部最优
   时间线中字典序最小的两条；两者 `cost` 相同。
6. **确定性**：同一请求体重复提交，响应**逐字节一致**（无时间戳、无随机、
   字典序固定）。
7. **错误优先级**：模式错误（类型/取值范围/多余字段）→ `VALIDATION_ERROR`；
   长度不符 → `LENGTH_MISMATCH`；余数越界 → `INVALID_REMAINDER`；
   起点不同余 → `INVALID_START`；`lo_i > hi_i` → `INVALID_INTERVAL`；
   候选数 > 64 → `TOO_MANY_CANDIDATES`。所有非法输入（422）优先于
   无解判定（200 + `impossible`）：只要请求本身非法，一律返回结构化错误。

## 4. API

### `POST /solve`

请求体（JSON，所有整数必须是 JSON 整数；浮点、布尔、字符串一律拒绝）：

```json
{
  "M": 5,                 // 模，2 ≤ M ≤ 1e9
  "n": 3,                 // 历元数，1 ≤ n ≤ 20000
  "r": [0, 1, 2],         // 余数，长度 n，0 ≤ r_i < M
  "x0": 0,                // 起点，x0 ≡ r_0 (mod M)，可为负
  "lo": [1, 6],           // 增量下界，长度 n-1
  "hi": [1, 6]            // 增量上界，长度 n-1，lo_i ≤ hi_i
}
```

响应（均 HTTP 200，`status` 三选一）：

```json
// 唯一
{"status": "unique", "x": [0, 1, 7], "cost": 5, "increments": [1, 6]}

// 不唯一：按 x 字典序最小的两条最优见证
{"status": "ambiguous", "cost": 0,
 "witnesses": [{"x": [0, 1, 2], "increments": [1, 1], "cost": 0},
               {"x": [0, 6, 12], "increments": [6, 6], "cost": 0}]}

// 无解：首个无候选时刻（时间下标 i，1 ≤ i ≤ n−1）
{"status": "impossible", "first_empty_index": 2}
```

错误（HTTP 422，结构化信封，`index` 为时间下标：余数用 `0..n-1`，
区间/候选用 `1..n-1`）：

```json
{"error": {"code": "INVALID_INTERVAL", "message": "...", "index": 2}}
```

| code | 含义 |
| --- | --- |
| `VALIDATION_ERROR` | JSON 模式错误：类型不符（浮点/布尔/字符串）、`M`/`n` 越界、缺字段、多余字段 |
| `LENGTH_MISMATCH` | `len(r) ≠ n` 或 `len(lo)/len(hi) ≠ n−1` |
| `INVALID_REMAINDER` | 某 `r_i` 不满足 `0 ≤ r_i < M` |
| `INVALID_START` | `x0 ≢ r_0 (mod M)` |
| `INVALID_INTERVAL` | 某 `lo_i > hi_i` |
| `TOO_MANY_CANDIDATES` | 某区间模差候选数 > 64 |

### `GET /health`

返回 `{"status": "ok"}`，供容器健康检查使用。

## 5. 运行

### Docker Compose（推荐）

```bash
docker compose up --build api          # 启动 API，宿主端口默认 8000
API_PORT=9000 docker compose up --build api   # 用 API_PORT 覆盖宿主端口
docker compose up --build verify       # 一次性验收：pytest + 对 API 的 HTTP 验收
docker compose down
```

- `api`：FastAPI 服务，容器内固定 8000 端口，宿主端口由 `API_PORT`
  环境变量覆盖（缺省 8000）。
- `verify`：一次性验收服务，等待 `api` 健康后先跑完整 pytest 套件，
  再对运行中的 API 做黑盒验收（已知答案、随机用例暴力对拍、
  大规模不变量与 `O(n·K²)` 代价复算、错误信封、逐字节确定性），
  全部通过则以退出码 0 结束，否则非 0。

### 本地开发

```bash
pip install -r requirements.txt
pytest                                   # 单元测试（含暴力枚举对拍与规模上限测试）
uvicorn app.main:app --reload --port 8000
API_BASE_URL=http://127.0.0.1:8000 python -m scripts.verify   # 端到端验收
```

## 6. 项目结构

```
app/
  main.py       FastAPI 应用：路由、结构化错误处理
  schemas.py    请求模式（strict 模式，拒绝浮点/布尔/字符串与多余字段）
  solver.py     纯整数求解器：校验、候选生成、O(n·K) DP、计数截断、见证重构
  errors.py     DomainError 与错误码
tests/
  brute.py      独立暴力枚举参考实现（itertools.product）
  test_solver.py  已知答案、随机对拍、转移算子对拍、规模上限
  test_api.py     HTTP 契约、错误信封、逐字节确定性
scripts/
  oracle.py     验收用独立暴力预言机
  acceptance.py 黑盒 HTTP 验收
  verify.py     一次性验收入口（pytest + HTTP 验收）
Dockerfile          python:3.13-slim 镜像
docker-compose.yml  api（API_PORT 覆盖宿主端口）+ verify（一次性验收）
requirements.txt    运行与测试依赖
```

## 7. 规模与性能

- 上限：`M ≤ 10^9`、`n ≤ 20000`、每时刻候选 ≤ 64。
- 复杂度：时间 `O(n·K)`（L1 距离变换两遍扫描），空间 `O(n·K)`
  （保存 `g` 行用于见证重构）；`n = 20000, K = 64` 实测约 0.5 秒、
  内存约百兆量级。
- 数值安全：全程 Python 任意精度整数，无浮点、无溢出；`x0` 与区间端点
  不限制取值范围（任意整数，含负数）。
