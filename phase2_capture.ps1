# ===== phase2_capture.ps1 =====
# 5x3 = 15 点 / 间距 0.8 m / 跨日 session 分目录
#
# 用法：
#   .\phase2_capture.ps1 -Session d1  -Round 1
#   .\phase2_capture.ps1 -Session d3  -Round 1
#   .\phase2_capture.ps1 -Session smoke -Round 1 -Only 0    # 冒烟测试：只采 label 0
#   .\phase2_capture.ps1 -Session smoke -Round 1 -Only 12   # 冒烟测试：再采一个第三排的点
#   .\phase2_capture.ps1 -Session d1  -Round 3 -Only 7 -Force         # 事后单点重采（会覆盖该点）
#
# 输出：<仓库>\data\<Session>\phase2_r<Round>\label<N>_pi<1|2>.pcap
#       每个 session 目录下另写 session_meta.json（UTF-8 带 BOM，Python 读取要用 encoding="utf-8-sig"）
#
# 2026-10-07 修订：
#   - -Session 不再有默认值，必须显式给出（防止 D+3 忘写 session 覆盖 D1）
#   - 目标轮次目录里已有 pcap 时拒绝运行，除非加 -Force
#   - 新增 -Only <label>：只采一个点（冒烟测试第三排 / 单点重采）
#   - 采集失败时默认重采，要跳过必须输入 skip；跳过的点记入 meta
#   - scp 前先删本地旧文件，避免重采失败时旧文件冒充新数据
#   - meta 每轮额外记录采集秒数、Pi 开机时间（uptime -s，重启会重置 CFO/AGC）、结束时间
#   - 每个点（含每次重采）在采集前后各记一次两台 Pi 的 SoC 温度、降频状态、CPU 频率，
#     写入 phase2_r<R>\points_meta.csv（一行一次尝试，status 列标明 ok/low/under/fail）
#   - 改为按**时间**采集（-Seconds，默认 42 s），不再按总包数（-Pkts 已删除）。
#     按总包数时 piA 因邻居路由器的 CSI 14 s 就满 600 包，piB 要 29 s；按时间则两台一致。
#     两台 AP 的 beacon 间隔 102.4 ms ≈ 9.77 帧/秒/链路，42 s ≈ 410 帧/链路（目标 ≥ 400）。
#     需要 Pi 上是新版 csi_capture.sh（含 CAPTURE_MODE_SECONDS），开采前自动检查。
#
# 2026-10-11 修订（D+3 前）：所有 ssh/scp 加执行超时。
#   起因：D1 r10 label4 采完后，读 piB 温度的 ssh 连上后挂住，脚本无限等待。
#   - 同步调用的 ssh/scp 全部改走 Invoke-Native：本地硬超时，到时直接杀掉 ssh 进程，脚本继续
#   - 所有连接加 ServerAliveInterval=5 / ServerAliveCountMax=3：网络断了 15 s 内自动断开
#   - Pi 端 vcgencmd 前加 timeout 5：命令本身卡住时尽量在远端就结束
#   - 采集本身（Start-Job + Wait-Job -Timeout）原来就有超时，只加了 keepalive 选项
#   当天 10:21 曾因怀疑本版导致 Pi 断流而退回原版；原版下 10:34 照样断流，排除了本脚本。
#
# 2026-10-11 晚修订（D+5 前）：恢复上面的超时版本，并在开采前检查两台 Pi 的无线配置。
#   D+3 上午三次断流的原因在 **Pi 的配置**（无线省电开着、NetworkManager 接管 wlan0），
#   不是本脚本的问题；修复在 Pi 端的 csi_setup.sh。这里只是开采前把关（和检查 csi_capture.sh
#   版本一样）：任一台 Pi 不是 Power Management:off 且 wlan0:unmanaged，就拒绝开采。

param(
    [ValidateRange(0, 999)]
    [int]$Round = 0,

    # 采集日标识：d1 / d3 / d5 / smoke。必填。不同日期必须用不同值，否则数据会混在一起。
    [ValidatePattern('^[a-z0-9_]+$')]
    [string]$Session = "",

    # 只采这一个 label（0..14）；默认 -1 = 采整轮 15 点
    [ValidateRange(-1, 14)]
    [int]$Only = -1,

    # 允许覆盖已存在的 pcap
    [switch]$Force,

    # 每个点采集的秒数。两台 AP 的 beacon ≈ 9.77 帧/秒/链路，42 s ≈ 410 帧/链路。
    [ValidateRange(5, 300)]
    [int]$Seconds = 42,

    # 本轮的环境备注，会写进 session_meta.json（例如 "门关 窗关 椅子原位"）
    [string]$RoomNote = "",

    # 两台接收端的 SSH 地址。可传参数，也可设置 CSI_PIA_HOST / CSI_PIB_HOST。
    [string]$PiAHost = $env:CSI_PIA_HOST,
    [string]$PiBHost = $env:CSI_PIB_HOST,

    # 两台 AP 的实际 BSSID。仅用于现场分链路和计数，不写入仓库。
    [string]$ApXBssid = $env:CSI_APX_BSSID,
    [string]$ApYBssid = $env:CSI_APY_BSSID,

    [string]$SshUser = "pi",

    # 默认写入仓库根目录下的 data；需要外置磁盘时可显式覆盖。
    [string]$OutputRoot = ""
)

# --- config ---
$Pi1          = $PiAHost.Trim()
$Pi2          = $PiBHost.Trim()
$ApXBssid     = $ApXBssid.Trim()
$ApYBssid     = $ApYBssid.Trim()
$RemoteScript = "/home/pi/csi_capture.sh"
$RemoteDir    = "/tmp"
$OutRoot      = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Join-Path $PSScriptRoot "data"
} else {
    $OutputRoot
}
$CapTimeout   = $Seconds + 30   # 秒。采集时长 + 预检/解析的余量
$Countdown    = 8

# 每条链路的帧数阈值（目标 400 帧/链路，留 5% 余量）
$ThGreen      = 380
$ThYellow     = 300

# --- grid config (change only this block when scene changes) ---
$Grid = @{
    Width   = 5
    Height  = 3
    Spacing = 0.8
    Path    = "snake"
}

# 蛇形走位，label = y*Width + x
#   y=0: 0 -> 4      (x 正向)
#   y=1: 9 -> 5      (x 反向，掉头)
#   y=2: 10 -> 14    (x 正向)
$WalkOrder = @(
    @{X=0;Y=0}, @{X=1;Y=0}, @{X=2;Y=0}, @{X=3;Y=0}, @{X=4;Y=0},
    @{X=4;Y=1}, @{X=3;Y=1}, @{X=2;Y=1}, @{X=1;Y=1}, @{X=0;Y=1},
    @{X=0;Y=2}, @{X=1;Y=2}, @{X=2;Y=2}, @{X=3;Y=2}, @{X=4;Y=2}
)

# 设备坐标，按 capture\摆位实施SOP.md，写进 meta 供论文引用
$Devices = @{
    apX = @{ x = 0.75;  y = -0.70; h = 0.75 }
    apY = @{ x = 4.10;  y = -0.30; h = 1.35 }
    piA = @{ x = -0.90; y = 2.00;  h = 1.10 }
    piB = @{ x = 2.65;  y = 2.20;  h = 0.60 }
}

# --- ssh/scp 执行超时（2026-10-11）---
$SshExe  = (Get-Command ssh.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$ScpExe  = (Get-Command scp.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$SshOpts = @('-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3')

# 运行 ssh/scp，超过 $TimeoutSec 秒就杀掉进程。输出按行返回（调用处用 @() 包起来）；超时或无输出时什么都不返回。
# 参数里含空格或分号的会加双引号；参数里不能有双引号（本脚本没有）。
function Invoke-Native {
    param([string]$Exe, [string[]]$ArgList, [int]$TimeoutSec = 15, [string]$What = "ssh")
    $quoted = foreach ($a in $ArgList) { if ($a -match '[\s;]') { '"' + $a + '"' } else { $a } }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $Exe
    $psi.Arguments              = ($quoted -join ' ')
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardInput  = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardInput.Close()                          # 相当于 ssh -n，不吃控制台输入
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $null    = $proc.StandardError.ReadToEndAsync()      # 必须读走 stderr，否则缓冲区满了会卡住
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { $proc.Kill() } catch {}
        Write-Host ("  !! {0}: {1} s 未返回，已强制结束 ssh（脚本继续）" -f $What, $TimeoutSec) -ForegroundColor Red
        return
    }
    $proc.WaitForExit()
    $outTask.Result -split "`r?`n" | Where-Object { $_ -ne "" }
}

function Show-Link {
    param([string]$Name, [int]$Count)
    if ($Count -ge $ThGreen) {
        Write-Host ("    {0} = {1}  OK" -f $Name, $Count) -ForegroundColor Green
    } elseif ($Count -ge $ThYellow) {
        Write-Host ("    {0} = {1}  LOW (kept)" -f $Name, $Count) -ForegroundColor Yellow
    } else {
        Write-Host ("    {0} = {1}  UNDER (redo!)" -f $Name, $Count) -ForegroundColor Red
    }
}

function Write-SessionMeta {
    param([string]$SessionDir, [int]$Round, [string]$Note)
    $metaPath = Join-Path $SessionDir "session_meta.json"
    if (Test-Path $metaPath) {
        $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    } else {
        $meta = [ordered]@{
            session      = $Session
            grid         = $Grid
            devices      = $Devices
            seconds_per_pt = $Seconds
            rounds       = @()
        }
    }
    $entry = [ordered]@{
        round         = $Round
        only_label    = $(if ($Only -ge 0) { $Only } else { $null })
        seconds       = $Seconds
        started_utc   = (Get-Date).ToUniversalTime().ToString("o")
        finished_utc  = $null
        room_note     = $Note
        pi_temp       = @{}
        pi_boot_time  = @{}
        skipped_labels= @()
    }
    foreach ($p in @(@("piA",$Pi1), @("piB",$Pi2))) {
        $r = @(Invoke-Native $SshExe ($SshOpts + @("$SshUser@$($p[1])", "timeout 5 vcgencmd measure_temp; uptime -s")) 12 "读 $($p[0]) 温度/开机时间")
        # 按内容认行，不按位置：vcgencmd 超时没输出时，uptime 那行不会被错当成温度
        $entry.pi_temp[$p[0]]      = "$(@($r | Where-Object { $_ -like 'temp=*' })[0])"
        $entry.pi_boot_time[$p[0]] = "$(@($r | Where-Object { $_ -match '^\d{4}-\d{2}-\d{2} ' })[0])"
    }
    $meta.rounds = @($meta.rounds) + $entry
    $meta | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $metaPath
    Write-Host "  meta -> $metaPath" -ForegroundColor DarkGray
    Write-Host ("  Pi temps: piA {0} | piB {1}" -f $entry.pi_temp.piA, $entry.pi_temp.piB) -ForegroundColor DarkGray
    Write-Host ("  Pi boot : piA {0} | piB {1}" -f $entry.pi_boot_time.piA, $entry.pi_boot_time.piB) -ForegroundColor DarkGray
}

# 本轮结束时补写结束时间和被跳过的点（更新最后一条 round 记录）
function Complete-SessionMeta {
    param([string]$SessionDir, [int[]]$Skipped)
    $metaPath = Join-Path $SessionDir "session_meta.json"
    if (-not (Test-Path $metaPath)) { return }
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    $last = @($meta.rounds)[-1]
    $last | Add-Member -NotePropertyName finished_utc   -NotePropertyValue (Get-Date).ToUniversalTime().ToString("o") -Force
    $last | Add-Member -NotePropertyName skipped_labels -NotePropertyValue @($Skipped) -Force
    $meta | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $metaPath
}

# 防覆盖：目标 pcap 已存在且没有 -Force 时返回 $false
function Test-NoOverwrite {
    param([string]$OutDir, [int[]]$Labels)
    if ($Force) { return $true }
    $existing = @()
    foreach ($l in $Labels) {
        foreach ($p in 1, 2) {
            $f = Join-Path $OutDir "label${l}_pi$p.pcap"
            if (Test-Path -LiteralPath $f) { $existing += (Split-Path $f -Leaf) }
        }
    }
    if ($existing.Count -eq 0) { return $true }
    Write-Host "`nREFUSED: $OutDir already has $($existing.Count) pcap file(s), e.g. $($existing[0])." -ForegroundColor Red
    Write-Host "  Did you forget to change -Session or -Round?" -ForegroundColor Red
    Write-Host "  If you really mean to overwrite, add -Force." -ForegroundColor Red
    return $false
}

# 一次 ssh 取一台 Pi 的状态：SoC 温度(°C)、降频标志(get_throttled)、CPU 频率(kHz)
#   throttled 非 0x0 表示欠压/过热降频（位 0 欠压、位 1 降频、位 2 限频、位 3 软温限）
function Get-PiStatus {
    param([string]$Ip)
    $r = @(Invoke-Native $SshExe ($SshOpts + @("$SshUser@$Ip",
           "timeout 5 vcgencmd measure_temp; timeout 5 vcgencmd get_throttled; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")) 15 "读 $Ip 温度")
    $txt = $r -join "`n"
    $st = @{ temp = ""; throttled = ""; freq_khz = "" }
    if ($txt -match 'temp=([\d.]+)')         { $st.temp = $Matches[1] }
    if ($txt -match 'throttled=(0x[0-9a-fA-F]+)') { $st.throttled = $Matches[1] }
    if ($txt -match '(?m)^(\d{5,})\s*$')      { $st.freq_khz = $Matches[1] }
    return $st
}

# 每次尝试追加一行到 phase2_r<R>\points_meta.csv（UTF-8 带 BOM，Python 用 utf-8-sig 读）
function Write-PointLog {
    param([string]$OutDir, $Row)   # $Row 用 [ordered]，不能声明成 [hashtable]，否则列顺序丢失
    $csv = Join-Path $OutDir "points_meta.csv"
    [PSCustomObject]$Row | Export-Csv -LiteralPath $csv -Append -NoTypeInformation -Encoding UTF8
}

# returns the minimum link count (or -1 if capture failed)
function Capture-Point {
    param(
        [Parameter(Mandatory)][int]$Round,
        [Parameter(Mandatory)][int]$X,
        [Parameter(Mandatory)][int]$Y,
        [Parameter(Mandatory)][string]$SessionDir,
        [int]$Attempt = 1
    )
    if ($X -lt 0 -or $X -ge $Grid.Width -or $Y -lt 0 -or $Y -ge $Grid.Height) {
        Write-Host "  X/Y out of range: grid is $($Grid.Width)x$($Grid.Height), valid x in 0..$($Grid.Width-1) y in 0..$($Grid.Height-1)" -ForegroundColor Red
        return -1
    }
    $label  = $Y * $Grid.Width + $X
    $outDir = Join-Path $SessionDir "phase2_r$Round"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $remotePath = "$RemoteDir/r${Round}_l${label}.pcap"

    Write-Host "`n=== $Session  r$Round  label$label  (x=$X y=$Y)  attempt $Attempt ===" -ForegroundColor Cyan
    $preA = Get-PiStatus $Pi1
    $preB = Get-PiStatus $Pi2
    Write-Host ("  Pi temp before: piA {0} C | piB {1} C   throttled {2} / {3}" -f $preA.temp, $preB.temp, $preA.throttled, $preB.throttled) -ForegroundColor DarkGray
    for ($i = $Countdown; $i -gt 0; $i--) { Write-Host -NoNewline "`r  ready $i "; Start-Sleep 1 }
    Write-Host "`r  capturing ${Seconds}s - stand still...                  " -ForegroundColor Yellow
    $startUtc = (Get-Date).ToUniversalTime().ToString("o")

    $null = Invoke-Native $SshExe ($SshOpts + @("$SshUser@$Pi1", "rm -f $remotePath")) 10 "piA 删旧 pcap"
    $null = Invoke-Native $SshExe ($SshOpts + @("$SshUser@$Pi2", "rm -f $remotePath")) 10 "piB 删旧 pcap"

    $cmd = "timeout $CapTimeout bash $RemoteScript $X $Y $label $Seconds $remotePath $ApXBssid $ApYBssid"
    $sb  = { param($u,$h,$c) ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$u@$h" $c }
    $j1 = Start-Job $sb -ArgumentList $SshUser,$Pi1,$cmd
    $j2 = Start-Job $sb -ArgumentList $SshUser,$Pi2,$cmd
    Wait-Job $j1,$j2 -Timeout ($CapTimeout + 15) | Out-Null
    $o1 = (Receive-Job $j1) -join "`n"; $o2 = (Receive-Job $j2) -join "`n"
    Remove-Job $j1,$j2 -Force

    $c1a1 = $c1a2 = $c2a1 = $c2a2 = -1
    if ($o1 -match 'LINKCOUNT (\d+) (\d+)') { $c1a1 = [int]$Matches[1]; $c1a2 = [int]$Matches[2] }
    if ($o2 -match 'LINKCOUNT (\d+) (\d+)') { $c2a1 = [int]$Matches[1]; $c2a2 = [int]$Matches[2] }

    $ok = $true
    foreach ($pair in @(@(1,$Pi1),@(2,$Pi2))) {
        $p,$ip = $pair
        $dst = Join-Path $outDir "label${label}_pi$p.pcap"
        Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue   # 防止旧文件冒充本次数据
        $null = Invoke-Native $ScpExe ($SshOpts + @("${SshUser}@${ip}:$remotePath", $dst)) 30 "scp pi$p"
        if (-not (Test-Path $dst)) {
            Write-Host "  pi$p recover FAILED (remote timeout/undersampled)" -ForegroundColor Red
            $ok = $false
        }
    }

    $endUtc = (Get-Date).ToUniversalTime().ToString("o")
    $postA = Get-PiStatus $Pi1
    $postB = Get-PiStatus $Pi2

    Write-Host "  4-link counts:"
    Show-Link "piA-apX" $c1a1
    Show-Link "piA-apY" $c1a2
    Show-Link "piB-apX" $c2a1
    Show-Link "piB-apY" $c2a2
    Write-Host ("  Pi temp after : piA {0} C | piB {1} C   throttled {2} / {3}" -f $postA.temp, $postB.temp, $postA.throttled, $postB.throttled) -ForegroundColor DarkGray
    foreach ($st in @(@("piA",$postA), @("piB",$postB))) {
        if ($st[1].throttled -and $st[1].throttled -ne "0x0") {
            Write-Host ("  !! {0} throttled={1}: 欠压或过热降频，检查电源/散热；本点已照常记录" -f $st[0], $st[1].throttled) -ForegroundColor Red
        }
    }

    $min = -1
    $validCounts = @($c1a1,$c1a2,$c2a1,$c2a2) | Where-Object { $_ -ge 0 }
    if ($ok -and $validCounts.Count -eq 4) {
        $min = ($validCounts | Measure-Object -Minimum).Minimum
    }
    if ($min -lt 0) {
        $status = "fail"
    } elseif ($min -ge $ThGreen) {
        $status = "ok";    Write-Host "  => all links OK" -ForegroundColor Green
    } elseif ($min -ge $ThYellow) {
        $status = "low";   Write-Host "  => usable but some low" -ForegroundColor Yellow
    } else {
        $status = "under"; Write-Host "  => UNDER threshold" -ForegroundColor Red
    }

    Write-PointLog -OutDir $outDir -Row ([ordered]@{
        session = $Session; round = $Round; label = $label; x = $X; y = $Y; attempt = $Attempt
        status = $status; seconds = $Seconds; start_utc = $startUtc; end_utc = $endUtc
        cnt_piA_apX = $c1a1; cnt_piA_apY = $c1a2; cnt_piB_apX = $c2a1; cnt_piB_apY = $c2a2
        piA_temp_pre = $preA.temp; piA_temp_post = $postA.temp
        piB_temp_pre = $preB.temp; piB_temp_post = $postB.temp
        piA_throttled_pre = $preA.throttled; piA_throttled_post = $postA.throttled
        piB_throttled_pre = $preB.throttled; piB_throttled_post = $postB.throttled
        piA_freq_khz = $postA.freq_khz; piB_freq_khz = $postB.freq_khz
    })
    return $min
}

function Capture-Round {
    param([Parameter(Mandatory)][int]$Round)

    $sessionDir = Join-Path $OutRoot $Session
    $outDir     = Join-Path $sessionDir "phase2_r$Round"

    # 要采的点：整轮按蛇形顺序；-Only 时只采一个
    if ($Only -ge 0) {
        $points = @(@{ X = $Only % $Grid.Width; Y = [int][math]::Floor($Only / $Grid.Width) })
    } else {
        $points = $WalkOrder
    }
    $labels = @($points | ForEach-Object { $_.Y * $Grid.Width + $_.X })

    # 防覆盖检查放在写 meta 之前，被拒绝时什么都不改
    if (-not (Test-NoOverwrite -OutDir $outDir -Labels $labels)) { exit 1 }

    # 两台 Pi 上必须是按时间采集的新版 csi_capture.sh；旧版会把秒数当包数，结果全错
    foreach ($ip in @($Pi1, $Pi2)) {
        $v = @(Invoke-Native $SshExe ($SshOpts + @("$SshUser@$ip", "grep -c CAPTURE_MODE_SECONDS $RemoteScript")) 10 "检查 $ip 脚本版本") -join ""
        if ("$v".Trim() -eq "" -or [int]("$v".Trim()) -lt 1) {
            Write-Host "`nREFUSED: $ip 上的 $RemoteScript 是旧版（按包数采集）或连不上。" -ForegroundColor Red
            Write-Host "  先把仓库里的 capture\pi\csi_capture.sh 拷过去：" -ForegroundColor Red
            Write-Host "  scp .\capture\pi\csi_capture.sh ${SshUser}@${ip}:$RemoteScript" -ForegroundColor Red
            exit 1
        }
    }

    # 两台 Pi 的无线配置必须是"省电关 + NetworkManager 不接管 wlan0"（见文件头 2026-10-11 晚修订）。
    # 配置不对时修 Pi，不修脚本：在该 Pi 上重跑新版 csi_setup.sh。
    foreach ($ip in @($Pi1, $Pi2)) {
        $chk = @(Invoke-Native $SshExe ($SshOpts + @("$SshUser@$ip",
                 "/sbin/iwconfig wlan0 2>/dev/null | grep -o 'Power Management:[a-z]*'; nmcli -t -f DEVICE,STATE dev status 2>/dev/null | grep '^wlan0:'")) 10 "检查 $ip 无线配置") -join " "
        if ($chk -notmatch 'Power Management:off' -or $chk -notmatch 'wlan0:unmanaged') {
            Write-Host "`nREFUSED: $ip 的无线配置不对：[$chk]" -ForegroundColor Red
            Write-Host "  需要 Power Management:off 且 wlan0:unmanaged。在该 Pi 上重跑新版 setup：" -ForegroundColor Red
            Write-Host "  ssh ${SshUser}@${ip} `"bash /home/pi/csi_setup.sh`"" -ForegroundColor Red
            exit 1
        }
        Write-Host "  $ip 无线配置 OK：$chk" -ForegroundColor DarkGray
    }

    New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null
    $total   = $points.Count
    $skipped = @()

    Write-Host "`n########################################" -ForegroundColor Magenta
    Write-Host "###  SESSION $Session  ROUND $Round  -  $total point(s)  ###" -ForegroundColor Magenta
    if ($Only -ge 0) { Write-Host "###  ONLY label $Only  ###" -ForegroundColor Magenta }
    Write-Host "###  RE-ENTER the area and STAND FRESH  ###" -ForegroundColor Magenta
    Write-Host "########################################" -ForegroundColor Magenta

    Write-SessionMeta -SessionDir $sessionDir -Round $Round -Note $RoomNote

    for ($idx = 0; $idx -lt $total; $idx++) {
        $x = $points[$idx].X
        $y = $points[$idx].Y
        $label = $y * $Grid.Width + $x

        $attempt = 0
        while ($true) {
            $attempt++
            Write-Host "`n----------------------------------------" -ForegroundColor White
            Write-Host ("NEXT [{0}/{1}]  label {2}  ->  stand at (x={3}, y={4})" -f ($idx+1), $total, $label, $x, $y) -ForegroundColor Cyan
            Read-Host "Press ENTER when you are in position"

            $min = Capture-Point -Round $Round -X $x -Y $y -SessionDir $sessionDir -Attempt $attempt

            if ($min -ge $ThGreen) {
                break
            } elseif ($min -ge $ThYellow) {
                # 偏低但可用：默认保留
                $ans = Read-Host "  some link LOW (min=$min). r = redo / ENTER = keep and continue"
                if ($ans -eq 'r' -or $ans -eq 'R') { continue } else { break }
            } else {
                # 失败或低于下限：默认重采，跳过必须明确输入 skip
                $ans = Read-Host "  FAILED / UNDER (min=$min). ENTER = redo / type 'skip' = leave this point"
                if ($ans -eq 'skip') {
                    $skipped += $label
                    Write-Host "  label $label SKIPPED - recorded in session_meta.json" -ForegroundColor Red
                    break
                }
                continue
            }
        }
    }

    Complete-SessionMeta -SessionDir $sessionDir -Skipped $skipped

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "###  $Session  ROUND $Round  DONE  ###" -ForegroundColor Green
    if ($skipped.Count -gt 0) {
        Write-Host ("###  SKIPPED labels: {0}  -> redo with -Only <label> -Force" -f ($skipped -join ', ')) -ForegroundColor Red
    }
    Write-Host "========================================" -ForegroundColor Green
}

if ($Round -gt 0) {
    if ([string]::IsNullOrEmpty($Session)) {
        Write-Host "ERROR: -Session is required (d1 / d3 / d5 / smoke). Refusing to guess." -ForegroundColor Red
        exit 1
    }
    if ([string]::IsNullOrWhiteSpace($Pi1) -or [string]::IsNullOrWhiteSpace($Pi2)) {
        Write-Host "ERROR: receiver addresses are required. Set CSI_PIA_HOST and CSI_PIB_HOST, or pass -PiAHost and -PiBHost." -ForegroundColor Red
        exit 1
    }
    $bssidPattern = '^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$'
    if ($ApXBssid -notmatch $bssidPattern -or $ApYBssid -notmatch $bssidPattern) {
        Write-Host "ERROR: valid AP BSSIDs are required. Set CSI_APX_BSSID and CSI_APY_BSSID, or pass -ApXBssid and -ApYBssid." -ForegroundColor Red
        exit 1
    }
    Capture-Round -Round $Round
} else {
    Write-Host "Loaded. Grid $($Grid.Width)x$($Grid.Height) @ $($Grid.Spacing) m, $($WalkOrder.Count) points."
    Write-Host "Set:    `$env:CSI_PIA_HOST='<piA-address>'; `$env:CSI_PIB_HOST='<piB-address>'"
    Write-Host "Start:  .\phase2_capture.ps1 -Session d1 -Round 1 -RoomNote `"...`""
    Write-Host "Smoke:  .\phase2_capture.ps1 -Session smoke -Round 1 -Only 0"
    Write-Host "        .\phase2_capture.ps1 -Session smoke -Round 1 -Only 12"
    Write-Host "Redo 1: .\phase2_capture.ps1 -Session d1 -Round 3 -Only 7 -Force"
}
