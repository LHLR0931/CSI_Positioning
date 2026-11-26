#!/usr/bin/env bash
# csi_capture.sh —— 在树莓派上采一个格点的 CSI（由笔记本上的 phase2_capture.ps1 通过 ssh 调用）
#
# 2026-10-07 起：按**时间**采集（CAPTURE_MODE_SECONDS），不再按总包数。
#   原因：按总包数时，piA 收到大量邻居路由器的 CSI，14 s 就满 600 包；piB 几乎收不到，
#   要 29 s。两台 Pi 时长差一倍，而且随邻居流量变化。我们两台 AP 的 beacon 都是
#   每 102.4 ms 一帧（≈9.8 帧/秒/链路），所以固定时长 = 固定帧数，与邻居流量无关。
#
# 用法: csi_capture.sh <x> <y> <label> <seconds> <out_pcap_abspath> <apX_bssid> <apY_bssid>
# 最后一行输出: LINKCOUNT <apX帧数> <apY帧数>    （phase2_capture.ps1 靠这一行判断）
# 版本标记（phase2_capture.ps1 开采前会检查，旧脚本会被拒绝）: CAPTURE_MODE_SECONDS
IFACE="wlan0"
CHANSPEC="36/20"
X="$1"; Y="$2"; LABEL="$3"; SECONDS_TO_CAPTURE="$4"; OUT="$5"
APX_BSSID="$6"; APY_BSSID="$7"

if [ -z "$X" ] || [ -z "$Y" ] || [ -z "$LABEL" ] || [ -z "$SECONDS_TO_CAPTURE" ] || [ -z "$OUT" ] || [ -z "$APX_BSSID" ] || [ -z "$APY_BSSID" ]; then
  echo "usage: $0 <x> <y> <label> <seconds> <out_pcap_abspath> <apX_bssid> <apY_bssid>"; exit 1
fi
case "$SECONDS_TO_CAPTURE" in
  ''|*[!0-9]*) echo "seconds must be a positive integer, got '$SECONDS_TO_CAPTURE'"; exit 1;;
esac

bssid_discriminator() {
  local compact
  compact=$(printf '%s' "$1" | tr -d ':-' | tr '[:upper:]' '[:lower:]')
  if [[ ! "$compact" =~ ^[0-9a-f]{12}$ ]]; then
    echo "invalid BSSID" >&2
    return 1
  fi
  printf '%s %s' "${compact:4:4}" "${compact:8:4}"
}
AP1=$(bssid_discriminator "$APX_BSSID") || exit 1
AP2=$(bssid_discriminator "$APY_BSSID") || exit 1

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
echo "point (x=$X y=$Y) -> label=$LABEL, ${SECONDS_TO_CAPTURE}s, out=$OUT"

# 预检：3 秒内至少要有 5 个 CSI 包，否则 CSI 没配好，直接退出，不白等
CNT=$(sudo timeout 3 tcpdump -i $IFACE dst port 5500 -c 5 2>/dev/null | wc -l)
if [ "$CNT" -eq 0 ]; then echo "NO packets, abort"; exit 2; fi

echo "capturing ${SECONDS_TO_CAPTURE}s ..."
# -U：每个包立即写盘；timeout 发 SIGINT，tcpdump 会正常收尾、关闭 pcap 并打印统计
# 2>&1：tcpdump 的提示信息走 stdout，PowerShell 不再显示成红色 NativeCommandError
sudo timeout --signal=INT "$SECONDS_TO_CAPTURE" \
  tcpdump -i $IFACE dst port 5500 -U -w "$OUT" -Z "$USER" 2>&1
GOT=$(sudo tcpdump -r "$OUT" 2>/dev/null | wc -l)

HEX=$(sudo tcpdump -r "$OUT" -x 2>/dev/null)
N_AP1=$(echo "$HEX" | grep -c "$AP1")
N_AP2=$(echo "$HEX" | grep -c "$AP2")

META="$(dirname "$OUT")/meta.csv"
[ -e "$META" ] || echo "timestamp,label,x,y,seconds,total,n_ap1,n_ap2,chanspec,file" > "$META"
echo "$(date '+%F %T'),$LABEL,$X,$Y,$SECONDS_TO_CAPTURE,$GOT,$N_AP1,$N_AP2,$CHANSPEC,$(basename "$OUT")" >> "$META"

echo "done: label=$LABEL seconds=$SECONDS_TO_CAPTURE total=$GOT AP1=$N_AP1 AP2=$N_AP2"
echo "LINKCOUNT $N_AP1 $N_AP2"
