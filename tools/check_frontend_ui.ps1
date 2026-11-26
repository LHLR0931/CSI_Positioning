param(
    [string]$LogName = 'v090-editor-preflight.log',
    [string]$UnityEditorPath = $env:UNITY_EDITOR_PATH
)
$ErrorActionPreference = 'Stop'
$repositoryPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$projectPath = Join-Path $repositoryPath 'unity-app'
$editorPath = $UnityEditorPath
$logDirectory = Join-Path $projectPath 'Logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ([IO.Path]::GetFileName($LogName))
if ([string]::IsNullOrWhiteSpace($editorPath) -or -not (Test-Path -LiteralPath $editorPath)) {
    throw 'Set UNITY_EDITOR_PATH or pass -UnityEditorPath with the Unity.exe location.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectPath 'ProjectSettings\ProjectVersion.txt'))) { throw 'Unity project is missing beside tools.' }
if (Get-Process Unity -ErrorAction SilentlyContinue) { throw 'Close the existing Unity Editor before this check.' }
$editorArgs = @('-batchmode', '-quit', '-buildTarget', 'Android', '-projectPath', ('"' + $projectPath + '"'),
    '-executeMethod', 'CsiApp.Editor.FrontendV090Checks.RunUnity', '-logFile', ('"' + $logPath + '"'))
$checkProcess = Start-Process -FilePath $editorPath -ArgumentList $editorArgs -WindowStyle Hidden -PassThru
Write-Output ('Unity PID=' + $checkProcess.Id + ' Log=' + $logPath)
$checkProcess.WaitForExit()
$checkProcess.Refresh()
if ($checkProcess.ExitCode -ne 0 -or (Get-Content -LiteralPath $logPath -Raw) -notmatch 'CSI_FRONTEND_V090_PASS') {
    Get-Content -LiteralPath $logPath -Tail 35
    throw 'UI checks failed. Inspect the log.'
}
Write-Output 'CSI_FRONTEND_UI_CHECK_PASS'
