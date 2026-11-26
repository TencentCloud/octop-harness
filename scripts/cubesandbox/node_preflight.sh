#!/usr/bin/env bash
# CubeSandbox 节点前置检查：PVM 内核、KVM、XFS、时钟同步、控制面连通性。
#
# 用法（在每台节点上以 root 执行）：
#   sudo bash node_preflight.sh                          # 只查本机
#   sudo bash node_preflight.sh --control-ip 10.0.0.2    # 额外验证到控制节点的 3010 / 9000
#   sudo bash node_preflight.sh --allow-standard-kernel  # 尚未装 PVM 内核时，跳过内核检查
#
# 退出码：0 全部通过；1 有失败项。
set -uo pipefail

CONTROL_IP=""
ALLOW_STANDARD=0
FAILED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --control-ip) CONTROL_IP="${2:-}"; shift 2 ;;
        --allow-standard-kernel) ALLOW_STANDARD=1; shift ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAILED=1; }
skip() { printf '[SKIP] %s\n' "$1"; }
info() { printf '[INFO] %s\n' "$1"; }

# 1. PVM 宿主内核
KERNEL="$(uname -r)"
if [[ "$KERNEL" == *".cubesandbox.pvm.host"* ]]; then
    pass "内核为 PVM 宿主内核: $KERNEL"
elif [[ "$ALLOW_STANDARD" == "1" ]]; then
    skip "内核不是 PVM 版本（已按参数跳过）: $KERNEL"
else
    fail "内核不是 PVM 版本: $KERNEL（需安装 kernel-release-* 并设为默认启动项后重启）"
fi

# 2. KVM 设备
if [[ -c /dev/kvm ]]; then
    pass "/dev/kvm 存在且为字符设备"
else
    fail "/dev/kvm 不存在：PVM 内核未生效或 kvm_pvm 未加载（modprobe kvm_pvm）"
fi

# 3. KVM 模块
MODULES="$( (command -v lsmod >/dev/null && lsmod) || true )"
if grep -q '^kvm_pvm' <<<"$MODULES"; then
    pass "kvm_pvm 已加载"
elif grep -qE '^kvm_(intel|amd)' <<<"$MODULES"; then
    pass "原生 KVM 模块已加载（非 PVM）"
else
    fail "没有 KVM 模块：modprobe kvm_pvm 后重试"
fi

# 4. /data 文件系统
if mountpoint -q /data 2>/dev/null; then
    FSTYPE="$(findmnt -no FSTYPE -T /data 2>/dev/null || echo unknown)"
    if [[ "$FSTYPE" == "xfs" ]]; then
        pass "/data 已挂载且为 XFS（CoW 快照依赖 reflink）"
    else
        fail "/data 是 $FSTYPE，CubeSandbox 需要 XFS（mkfs.xfs 后重新挂载）"
    fi
else
    fail "/data 未单独挂载：/data/cubelet 需要一块 XFS 数据盘"
fi

# 5. 磁盘余量
if [[ -d /data ]]; then
    AVAIL_G="$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc '0-9')"
    AVAIL_G="${AVAIL_G:-0}"
    if (( AVAIL_G >= 200 )); then
        pass "/data 可用 ${AVAIL_G}G"
    elif (( AVAIL_G >= 50 )); then
        info "/data 可用 ${AVAIL_G}G（满足下限 50G，多模板建议 200G+）"
    else
        fail "/data 可用 ${AVAIL_G}G < 50G"
    fi
fi

# 6. glibc
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0.0)"
if awk -v v="$GLIBC" 'BEGIN{exit !(v+0 >= 2.31)}'; then
    pass "glibc $GLIBC >= 2.31"
else
    fail "glibc $GLIBC < 2.31：官方二进制无法运行"
fi

# 7. CPU / 内存
if [[ "$(uname -m)" == "x86_64" ]] && ! grep -qw avx2 /proc/cpuinfo; then
    fail "x86_64 但缺少 AVX2：s3lvol 无法运行"
else
    pass "CPU 指令集满足要求（$(uname -m)）"
fi
CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)"
MEM_G="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
if (( MEM_G >= 8 )); then
    pass "规格 ${CORES}C/${MEM_G}G（s3lvol 自身约需 2C/23G）"
else
    fail "内存 ${MEM_G}G < 8G"
fi

# 8. 时钟同步（心跳窗口 15s）
if command -v chronyc >/dev/null 2>&1 && chronyc tracking >/dev/null 2>&1; then
    LEAP="$(chronyc tracking | awk -F': ' '/Leap status/{print $2}')"
    if [[ "$LEAP" == "Normal" ]]; then
        pass "chrony 已同步（Leap status: Normal）"
    else
        fail "chrony 未同步（Leap status: ${LEAP:-unknown}）：节点会被判不健康、模板停止分发"
    fi
else
    skip "未安装或未运行 chrony，请确认 systemd-timesyncd / ntpd 等其他方式已同步"
fi

# 9. Docker
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    pass "Docker 运行中"
else
    fail "Docker 未安装或未运行（MySQL/Redis/JuiceFS 依赖它）"
fi

# 10. 控制面连通性
if [[ -n "$CONTROL_IP" ]]; then
    for port in 3010 9000; do
        if timeout 5 bash -c "</dev/tcp/${CONTROL_IP}/${port}" 2>/dev/null; then
            pass "可连通控制节点 ${CONTROL_IP}:${port}"
        else
            fail "无法连通 ${CONTROL_IP}:${port}（安全组 / 路由 / 服务未启动）"
        fi
    done
else
    skip "未提供 --control-ip，跳过控制面连通性检查"
fi

echo
if (( FAILED )); then
    echo "结论：存在失败项，先修复再装 CubeSandbox 或建集群。"
    exit 1
fi
echo "结论：全部通过。"
