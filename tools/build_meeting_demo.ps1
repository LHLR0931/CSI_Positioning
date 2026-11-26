param(
    [string]$LogName = 'build-android-positioning-v0.9.0.log',
    [string]$UnityEditorPath = $env:UNITY_EDITOR_PATH
)
$ErrorActionPreference = 'Stop'
$repositoryPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$projectPath = Join-Path $repositoryPath 'unity-app'
$editorPath = $UnityEditorPath
$logDirectory = Join-Path $projectPath 'Logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ([System.IO.Path]::GetFileName($LogName))
if ([string]::IsNullOrWhiteSpace($editorPath) -or -not (Test-Path -LiteralPath $editorPath)) {
    throw 'Set UNITY_EDITOR_PATH or pass -UnityEditorPath with the Unity.exe location.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectPath 'ProjectSettings\ProjectVersion.txt'))) { throw 'Unity project is missing beside tools.' }
if (Get-Process Unity -ErrorAction SilentlyContinue) { throw 'Close the existing Unity Editor before this batch build.' }
$editorArgs = @('-batchmode', '-quit', '-buildTarget', 'Android',
    '-projectPath', ('"' + $projectPath + '"'),
    '-executeMethod', 'CsiApp.Editor.ArPrototypeBuild.BuildAndroidWithModel',
    '-logFile', ('"' + $logPath + '"'))
$buildProcess = Start-Process -FilePath $editorPath -ArgumentList $editorArgs -WindowStyle Hidden -PassThru
Write-Output ('Unity PID=' + $buildProcess.Id + ' Log=' + $logPath)
$buildProcess.WaitForExit()
$buildProcess.Refresh()
$buildCode = $buildProcess.ExitCode
Write-Output ('Unity exit=' + $buildCode)
if ($null -ne $buildCode -and $buildCode -ne 0) { exit $buildCode }
$logContents = Get-Content -LiteralPath $logPath -Raw
if ($logContents -notmatch 'CSI_MODEL_ANDROID_BUILD_RESULT: Succeeded') {
    throw ('Build did not report success. Inspect ' + $logPath)
}
Write-Output 'CSI_MEETING_APK_BUILD_PASS'
