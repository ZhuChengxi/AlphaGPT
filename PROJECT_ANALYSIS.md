# AlphaGPT 项目详细分析报告

> 本文档由代码分析自动生成，旨在帮助开发者快速理解项目架构和学习路径。

---

## 一、项目概述

**AlphaGPT** 是一个基于深度学习算法与符号回归的量化因子挖掘框架。项目通过构建非线性、高阶的特征组合，在加密货币市场（Solana 生态）以及中国 A 股市场进行量化交易。

### 核心定位

| 功能模块 | 描述 |
|----------|------|
| **因子挖掘** | 使用 Transformer + 符号回归自动发现 Alpha 因子 |
| **实盘交易** | 集成 Solana 链上交易执行（Jupiter DEX） |
| **可视化监控** | Streamlit 仪表板实时追踪策略表现 |

---

## 二、系统架构

```
AlphaGPT/
├── model_core/          # 核心算法模块
│   ├── alphagpt.py      # 模型定义（LoRD正则化、循环Transformer）
│   ├── engine.py        # 训练引擎
│   ├── vm.py            # 栈式虚拟机（执行因子公式）
│   ├── factors.py       # 特征工程
│   ├── ops.py           # 操作算子库
│   ├── backtest.py      # 回测评估
│   ├── data_loader.py   # 数据加载
│   └── config.py        # 模型配置
│
├── data_pipeline/       # 数据管道
│   ├── data_manager.py  # 管道协调器
│   ├── db_manager.py    # PostgreSQL 管理
│   ├── fetcher.py       # API 数据获取
│   ├── processor.py     # 数据清洗
│   ├── config.py        # 数据配置
│   └── providers/       # 数据源适配器
│       ├── base.py      # 抽象基类
│       ├── birdeye.py   # Birdeye API
│       └── dexscreener.py # DexScreener API
│
├── execution/           # 交易执行
│   ├── trader.py        # Solana 交易器
│   ├── jupiter.py       # Jupiter DEX 集成
│   ├── rpc_handler.py   # RPC 客户端
│   ├── utils.py         # 工具函数
│   └── config.py        # 执行配置
│
├── strategy_manager/    # 策略管理
│   ├── runner.py        # 主循环运行器
│   ├── portfolio.py     # 仓位管理
│   ├── risk.py          # 风险控制
│   └── config.py        # 策略配置
│
├── dashboard/           # Web 仪表板
│   ├── app.py           # Streamlit 应用
│   ├── data_service.py  # 数据服务
│   └── visualizer.py    # Plotly 图表
│
└── code/main.py         # A股回测脚本（独立版本）
```

---

## 三、核心算法详解

### 3.1 AlphaGPT 模型架构

模型位于 `model_core/alphagpt.py`，包含以下关键组件：

| 组件 | 类名 | 作用 |
|------|------|------|
| **循环变换器** | `LoopedTransformer` | 单层内多次迭代处理信息，增强表达能力 |
| **RMS归一化** | `RMSNorm` | 替代 LayerNorm，梯度更稳定 |
| **QK归一化** | `QKNorm` | 查询-键独立归一化，防止数值爆炸 |
| **SwiGLU激活** | `SwiGLU` | Swish-GLU 前馈网络，捕捉非线性关系 |
| **多任务头** | `MTPHead` | 动态加权多目标学习 |
| **LoRD正则化** | `NewtonSchulzLowRankDecay` | 低秩约束防止过拟合 |
| **稳定秩监控** | `StableRankMonitor` | 监控参数有效秩 |

### 3.2 LoRD 正则化原理

Low-Rank Decay (LoRD) 使用 Newton-Schulz 迭代计算注意力参数的最小奇异向量，对参数施加低秩约束：

```
Newton-Schulz 迭代: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
```

这种方法无需显式 SVD 分解，计算效率更高。

### 3.3 符号回归框架

模型生成的公式由 **特征** 和 **操作** 组成：

**特征空间（6维）**：

| ID | 名称 | 含义 |
|----|------|------|
| 0 | RET | 对数收益率 |
| 1 | LIQ | 流动性评分 |
| 2 | PRESSURE | 买卖压力 |
| 3 | FOMO | FOMO 加速度 |
| 4 | DEV | 泵偏离度 |
| 5 | VOL | 对数交易量 |

**操作算子（12种）**：

| ID | 名称 | 元数 | 功能 |
|----|------|------|------|
| 6 | ADD | 2 | 加法 |
| 7 | SUB | 2 | 减法 |
| 8 | MUL | 2 | 乘法 |
| 9 | DIV | 2 | 除法（带保护） |
| 10 | NEG | 1 | 取负 |
| 11 | ABS | 1 | 绝对值 |
| 12 | SIGN | 1 | 符号函数 |
| 13 | GATE | 3 | 条件选择 |
| 14 | JUMP | 1 | 异常检测（3σ跳跃） |
| 15 | DECAY | 1 | 加权衰减 |
| 16 | DELAY1 | 1 | 延迟一期 |
| 17 | MAX3 | 1 | 三期最大值 |

### 3.4 栈式虚拟机 (StackVM)

`model_core/vm.py` 实现了一个栈式虚拟机，用于执行生成的因子公式：

```python
# 执行流程示例
公式: [0, 1, 6]  # RET, LIQ, ADD
执行:
  1. 遇到 0 (RET) → push(特征张量[:, 0, :])
  2. 遇到 1 (LIQ) → push(特征张量[:, 1, :])
  3. 遇到 6 (ADD) → pop两次, 执行加法, push结果
结果: 栈顶为计算后的因子值
```

### 3.5 训练流程

```
┌─────────────┐
│  数据加载   │ CryptoDataLoader 从 PostgreSQL 读取 OHLCV
└──────┬──────┘
       ↓
┌─────────────┐
│  特征工程   │ FeatureEngineer 计算 6 维特征
└──────┬──────┘
       ↓
┌─────────────┐
│  公式生成   │ AlphaGPT 采样 8192 个公式序列
└──────┬──────┘
       ↓
┌─────────────┐
│  公式执行   │ StackVM 执行每个公式
└──────┬──────┘
       ↓
┌─────────────┐
│  回测评估   │ MemeBacktest 计算收益和风险
└──────┬──────┘
       ↓
┌─────────────┐
│  策略梯度   │ Policy Gradient 优化模型
└──────┬──────┘
       ↓
┌─────────────┐
│  LoRD正则   │ Newton-Schulz 低秩约束
└─────────────┘
```

---

## 四、数据流详解

### 4.1 数据获取

```
Birdeye API → 趋势代币 → 过滤(流动性/FDV) → OHLCV历史 → PostgreSQL
```

### 4.2 数据库表结构

**tokens 表**：
```sql
CREATE TABLE tokens (
    address TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    decimals INT,
    chain TEXT,
    last_updated TIMESTAMP
);
```

**ohlcv 表**：
```sql
CREATE TABLE ohlcv (
    time TIMESTAMP NOT NULL,
    address TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    liquidity DOUBLE PRECISION,
    fdv DOUBLE PRECISION,
    source TEXT,
    PRIMARY KEY (time, address)
);
```

---

## 五、交易执行流程

### 5.1 买入流程

```
检查余额 → Jupiter报价 → 构造交易 → 签名 → 发送RPC → 确认 → 更新仓位
```

### 5.2 卖出流程

```
查询代币余额 → 计算卖出数量 → Jupiter报价 → 执行交易 → 更新/关闭仓位
```

### 5.3 风控机制

| 机制 | 参数 | 说明 |
|------|------|------|
| 最大持仓 | 3 | 同时最多持有 3 个代币 |
| 止损 | -5% | 亏损超过 5% 全部卖出 |
| 止盈 | +10% | 盈利超过 10% 卖出 50%，剩余变"月球包" |
| 拖尾止损 | 5% 激活，3% 回撤 | 盈利超 5% 后启用，从最高点回撤 3% 卖出 |
| 流动性检查 | $5,000 | 流动性过低不交易 |
| 蜜罐检测 | Jupiter 报价 | 无法获取卖出报价则跳过 |

---

## 六、项目质量评价

### 6.1 优点 ✅

| 维度 | 评分 | 说明 |
|------|------|------|
| **创新性** | ⭐⭐⭐⭐⭐ | 首创 LoRD 正则化 + 符号回归的因子挖掘方法 |
| **完整性** | ⭐⭐⭐⭐ | 端到端流程：训练→数据→执行→监控 |
| **工程质量** | ⭐⭐⭐⭐ | 异步架构、类型提示、日志规范 |
| **模块化** | ⭐⭐⭐⭐⭐ | 清晰分层，低耦合设计 |
| **可扩展性** | ⭐⭐⭐⭐ | Provider 抽象、配置外置 |

**具体亮点**：

1. **Newton-Schulz 迭代的 LoRD 正则化**：独特的低秩约束方法，无需显式 SVD
2. **栈式虚拟机**：高效执行生成的因子公式，支持错误恢复和数值稳定处理
3. **多层风控**：止损/止盈/拖尾/蜜罐检测/流动性验证
4. **异步高并发**：`asyncio` + `aiohttp` 支持高效数据获取

### 6.2 不足 ⚠️

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| **测试覆盖** | 高 | 缺少单元测试和集成测试 |
| **错误处理** | 中 | 部分异常被静默吞掉 |
| **文档注释** | 中 | 核心算法缺少数学推导注释 |
| **硬编码** | 低 | 部分参数直接写在代码中 |
| **数据源单一** | 中 | 主要依赖 Birdeye API |

---

## 七、学习路径规划

### 阶段一：理解项目背景（1小时）

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 1 | `README.md` | 了解项目动机和整体定位 |
| 2 | `requirements.txt` | 熟悉技术栈依赖 |

### 阶段二：核心算法理解（4-6小时）⭐重点

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 3 | `model_core/config.py` | 理解模型配置参数 |
| 4 | `model_core/ops.py` | 掌握12种操作算子定义 |
| 5 | `model_core/factors.py` | 理解特征工程和因子计算 |
| 6 | `model_core/vm.py` | **核心！** 栈式虚拟机执行原理 |
| 7 | `model_core/alphagpt.py` | **核心！** 模型架构和 LoRD 正则化 |
| 8 | `model_core/backtest.py` | 回测评估逻辑 |
| 9 | `model_core/engine.py` | 训练流程整合 |

### 阶段三：数据流理解（2-3小时）

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 10 | `data_pipeline/config.py` | 数据源配置 |
| 11 | `data_pipeline/providers/base.py` | Provider 抽象接口 |
| 12 | `data_pipeline/providers/birdeye.py` | API 调用实现 |
| 13 | `data_pipeline/db_manager.py` | 数据库操作 |
| 14 | `data_pipeline/data_manager.py` | 管道协调器 |

### 阶段四：交易执行（2-3小时）

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 15 | `execution/config.py` | Solana 钱包配置 |
| 16 | `execution/jupiter.py` | Jupiter DEX 集成 |
| 17 | `execution/trader.py` | 买卖交易逻辑 |

### 阶段五：策略运行（2小时）

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 18 | `strategy_manager/config.py` | 策略参数 |
| 19 | `strategy_manager/portfolio.py` | 仓位管理 |
| 20 | `strategy_manager/risk.py` | 风控引擎 |
| 21 | `strategy_manager/runner.py` | **核心！** 主循环逻辑 |

### 阶段六：扩展阅读（1-2小时）

| 序号 | 文件 | 学习目标 |
|------|------|----------|
| 22 | `dashboard/app.py` | Streamlit 界面 |
| 23 | `code/main.py` | A股回测脚本（简化版） |

---

## 八、快速上手指南

### 8.1 环境配置

```bash
# 1. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入必要配置
```

### 8.2 数据库准备

```bash
# 创建 PostgreSQL 数据库
createdb crypto_quant

# 可选：安装 TimescaleDB 扩展
# CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 8.3 运行训练

```bash
# 运行因子挖掘训练
python -m model_core.engine
```

### 8.4 启动策略

```bash
# 启动实盘策略
python -m strategy_manager.runner
```

### 8.5 启动仪表板

```bash
# 启动 Web 监控
cd dashboard
streamlit run app.py
```

---

## 九、关键配置参数

### 模型配置 (`model_core/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 8192 | 每批采样公式数 |
| TRAIN_STEPS | 1000 | 训练迭代次数 |
| MAX_FORMULA_LEN | 12 | 最大公式长度 |
| BASE_FEE | 0.5% | 基础交易费率 |

### 策略配置 (`strategy_manager/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_OPEN_POSITIONS | 3 | 最大持仓数 |
| ENTRY_AMOUNT_SOL | 2.0 | 单次入场金额 |
| STOP_LOSS_PCT | -5% | 止损阈值 |
| TAKE_PROFIT_Target1 | +10% | 止盈阈值 |
| BUY_THRESHOLD | 0.85 | 买入信号阈值 |
| SELL_THRESHOLD | 0.45 | 卖出信号阈值 |

---

## 十、总结

AlphaGPT 是一个**工程完成度较高、算法创新性强**的量化因子挖掘项目。主要创新点在于将 Transformer + 符号回归与 LoRD 正则化相结合，自动发现可交易的 Alpha 因子。

**适合人群**：
- 量化策略研究员
- 深度学习 + 金融交叉领域学习者
- Solana DeFi 开发者

**学习预估时间**：约 **15-20 小时** 可完整理解整个项目。

---

*文档生成时间：2026年3月*
