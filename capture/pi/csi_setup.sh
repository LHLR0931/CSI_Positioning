#!/usr/bin/env bash
# csi_setup.sh —— 每次 Pi 开机（或重载 brcmfmac 驱动）后运行一次，让网卡开始输出 CSI
#
# 2026-10-11 修订（D+3 当天断流 3 次后加入，D+5 起使用）：
#   - 让 NetworkManager 放手 wlan0（nmcli dev set wlan0 managed no）
#   - 关闭无线省电（iwconfig wlan0 power off）。必须放在 ifconfig up 之后：
#     日志显示每次 ifconfig wlan0 up 内核都会自动重新打开省电（power save enabled）
#   依据：改之前 54 分钟内两台 Pi 共断流 3 次；改之后连续 84 分钟（75 个点）无断流。
#   断流时刻的日志里没有 NetworkManager / wpa_supplicant 的任何动作，所以更可能是省电模式，
#   但两项未单独检验，都保留。两项设置重启后失效，所以每次开机都要跑本脚本。
#   断流时的快速恢复（不用重启，约 20 秒）：
#     sudo modprobe -r brcmfmac_wcc; sudo modprobe -r brcmfmac; sleep 2; sudo modprobe brcmfmac; sleep 8
#     然后重跑本脚本。注意：驱动重载等同于网卡重启，之后按规则要重新标定。
CHANSPEC="36/20"
CORE_MASK="1"
NSS_MASK="1"
FILTER_MAC="0"
IFACE="wlan0"
MCP="/home/pi/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams"
mkdir -p "$HOME/csi_data"
sudo rfkill unblock wifi
sudo nmcli dev set $IFACE managed no 2>/dev/null || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true
sudo ifconfig $IFACE up
if [ "$FILTER_MAC" = "1" ]; then
  AP_MAC="${CSI_FILTER_BSSID:-${1:-}}"
  if [ -z "$AP_MAC" ]; then
    echo "single-link mode requires CSI_FILTER_BSSID or a BSSID argument" >&2
    exit 1
  fi
  PARAMS=$($MCP -c $CHANSPEC -C $CORE_MASK -N $NSS_MASK -m $AP_MAC)
  echo "MODE: single-link (filter $AP_MAC)"
else
  PARAMS=$($MCP -c $CHANSPEC -C $CORE_MASK -N $NSS_MASK)
  echo "MODE: dual-link (no MAC filter, split by BSSID post-capture)"
fi
echo "$PARAMS" > "$HOME/csi_data/params.txt"
echo "PARAMS=$PARAMS"
sudo nexutil -I$IFACE -s500 -b -l34 -v $PARAMS
# 省电必须在 ifconfig up 之后关
sudo /sbin/iwconfig $IFACE power off
# 自检：两行都要对，否则不要开采
echo "CHECK power : $(/sbin/iwconfig $IFACE 2>/dev/null | grep -o 'Power Management:[a-z]*')   (要 off)"
echo "CHECK nm    : $(nmcli -t -f DEVICE,STATE dev status 2>/dev/null | grep "^$IFACE:")   (要 unmanaged)"
echo "setup done"
