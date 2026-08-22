#!/usr/bin/env bash
# ============================================================
# RLHGNN 完整流水线启动脚本 (Linux)
# 功能：依次执行 5 个核心代码文件，一个跑完再跑下一个
# 用法：
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                    # 使用默认 Python
#   ./run_pipeline.sh -p /usr/bin/python3 # 指定 Python 路径
#   ./run_pipeline.sh -e myenv           # 激活 conda 环境
#   ./run_pipeline.sh --skip-data        # 跳过 data_process.py
# ============================================================

set -euo pipefail

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- 默认配置 ----
PYTHON="python"
CONDA_ENV=""
SKIP_DATA=false
SKIP_PRE=false
SKIP_ENV=false
SKIP_FINAL=false
SKIP_METRICS=false

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--python)
            PYTHON="$2"
            shift 2
            ;;
        -e|--conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-pre)
            SKIP_PRE=true
            shift
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --skip-final)
            SKIP_FINAL=true
            shift
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -p, --python PATH     指定 Python 解释器路径 (默认: python)"
            echo "  -e, --conda-env NAME  激活指定 conda 环境"
            echo "  --skip-data           跳过 data_process.py"
            echo "  --skip-pre            跳过 pre_main.py"
            echo "  --skip-env            跳过 env_train.py"
            echo "  --skip-final          跳过 final_main.py"
            echo "  --skip-metrics        跳过 metrics_final.py"
            echo "  -h, --help            显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "用法: $0 [-p PYTHON] [-e CONDA_ENV] [--skip-*]"
            exit 1
            ;;
    esac
done

# ---- 检查 Python ----
if ! command -v "$PYTHON" &> /dev/null; then
    echo -e "${RED}[错误] 找不到 Python: $PYTHON${NC}"
    echo "请安装 Python 或使用 -p 参数指定正确路径。"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1)
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  RLHGNN 训练流水线${NC}"
echo -e "${CYAN}  Python: $PYTHON_VERSION${NC}"
echo -e "${CYAN}  工作目录: $(pwd)${NC}"
echo -e "${CYAN}========================================${NC}"

# ---- 激活 conda 环境（如需） ----
if [[ -n "$CONDA_ENV" ]]; then
    echo -e "${YELLOW}[信息] 激活 conda 环境: $CONDA_ENV${NC}"
    # 尝试 source conda
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [[ -n "$CONDA_BASE" ]]; then
        # shellcheck disable=SC1090
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        # 直接用 conda run
        PYTHON="conda run -n $CONDA_ENV python"
    fi
fi

# ---- 检查是否在项目根目录 ----
if [[ ! -f "data_process.py" ]]; then
    echo -e "${YELLOW}[警告] 当前目录下未找到 data_process.py${NC}"
    echo -e "${YELLOW}请确保在 RLHGNN 项目根目录下运行此脚本。${NC}"
fi

# ---- 计时函数 ----
SCRIPT_NAME=""
START_TIME=0

start_stage() {
    SCRIPT_NAME="$1"
    START_TIME=$(date +%s)
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  [$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
    echo -e "${GREEN}  开始执行: $SCRIPT_NAME${NC}"
    echo -e "${GREEN}========================================${NC}"
}

end_stage() {
    local exit_code=$1
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - START_TIME ))
    local hours=$(( duration / 3600 ))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$(( duration % 60 ))

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  ✅ $SCRIPT_NAME 执行完毕${NC}"
        echo -e "${GREEN}  耗时: ${hours}h ${minutes}m ${seconds}s${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}  ❌ $SCRIPT_NAME 执行失败 (退出码: $exit_code)${NC}"
        echo -e "${RED}  耗时: ${hours}h ${minutes}m ${seconds}s${NC}"
        echo -e "${RED}========================================${NC}"
    fi
}

# ---- 创建日志目录 ----
mkdir -p logs

# ============================================================
# 步骤 1: data_process.py — 数据预处理
# ============================================================
if [[ "$SKIP_DATA" == false ]]; then
    start_stage "1/5: data_process.py (数据预处理)"
    if $PYTHON data_process.py 2>&1 | tee "logs/data_process.log"; then
        end_stage 0
    else
        end_stage $?
        echo -e "${RED}[错误] data_process.py 执行失败，终止流水线。${NC}"
        echo "如想跳过此步骤，请使用 --skip-data 参数重新运行。"
        exit 1
    fi
else
    echo -e "${YELLOW}[跳过] data_process.py${NC}"
fi

# ============================================================
# 步骤 2: pre_main.py — 预训练基础 GNN 模型
# ============================================================
if [[ "$SKIP_PRE" == false ]]; then
    start_stage "2/5: pre_main.py (预训练基础 GNN 模型)"
    if $PYTHON pre_main.py 2>&1 | tee "logs/pre_main.log"; then
        end_stage 0
    else
        end_stage $?
        echo -e "${RED}[错误] pre_main.py 执行失败，终止流水线。${NC}"
        echo "如想跳过此步骤，请使用 --skip-pre 参数重新运行。"
        exit 1
    fi
else
    echo -e "${YELLOW}[跳过] pre_main.py${NC}"
fi

# ============================================================
# 步骤 3: env_train.py — 训练 DQN 决策模型
# ============================================================
if [[ "$SKIP_ENV" == false ]]; then
    start_stage "3/5: env_train.py (训练 DQN 决策模型)"
    if $PYTHON env_train.py 2>&1 | tee "logs/env_train.log"; then
        end_stage 0
    else
        end_stage $?
        echo -e "${RED}[错误] env_train.py 执行失败，终止流水线。${NC}"
        echo "如想跳过此步骤，请使用 --skip-env 参数重新运行。"
        exit 1
    fi
else
    echo -e "${YELLOW}[跳过] env_train.py${NC}"
fi

# ============================================================
# 步骤 4: final_main.py — 训练最终预测模型
# ============================================================
if [[ "$SKIP_FINAL" == false ]]; then
    start_stage "4/5: final_main.py (训练最终预测模型)"
    if $PYTHON final_main.py 2>&1 | tee "logs/final_main.log"; then
        end_stage 0
    else
        end_stage $?
        echo -e "${RED}[错误] final_main.py 执行失败，终止流水线。${NC}"
        echo "如想跳过此步骤，请使用 --skip-final 参数重新运行。"
        exit 1
    fi
else
    echo -e "${YELLOW}[跳过] final_main.py${NC}"
fi

# ============================================================
# 步骤 5: metrics_final.py — 计算最终评估指标
# ============================================================
if [[ "$SKIP_METRICS" == false ]]; then
    start_stage "5/5: metrics_final.py (计算评估指标)"
    if $PYTHON metrics_final.py 2>&1 | tee "logs/metrics_final.log"; then
        end_stage 0
    else
        end_stage $?
        echo -e "${RED}[错误] metrics_final.py 执行失败。${NC}"
        echo "如想跳过此步骤，请使用 --skip-metrics 参数重新运行。"
        exit 1
    fi
else
    echo -e "${YELLOW}[跳过] metrics_final.py${NC}"
fi

# ---- 完成 ----
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🎉 流水线全部执行完毕！${NC}"
echo -e "${GREEN}  各步骤日志保存在 logs/ 目录下${NC}"
echo -e "${GREEN}========================================${NC}"
