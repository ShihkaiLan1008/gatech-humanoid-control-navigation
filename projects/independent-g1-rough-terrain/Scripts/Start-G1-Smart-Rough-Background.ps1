param([int]$NumEnvs = 500, [int]$MaxIterations = 2500)

$backgroundDir = 'E:\NVIDIA\smart-rough-background'
$experimentRoot = 'E:\NVIDIA\IsaacLab\logs\rsl_rl\g1_energy_smart_rough'
New-Item -ItemType Directory -Path $backgroundDir -Force | Out-Null

$existingPidFile = Join-Path $backgroundDir 'pid.txt'
if (Test-Path -LiteralPath $existingPidFile) {
    $existingPid = Get-Content -LiteralPath $existingPidFile -ErrorAction SilentlyContinue
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Output "Smart rough-terrain training is already running (PID $existingPid)."
        exit 0
    }
}

$checkpoint = Get-ChildItem -LiteralPath $experimentRoot -Recurse -Filter 'model_*.pt' |
    Where-Object { $_.Directory.Name -ne 'seed_from_energy_medium' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $checkpoint) { throw "No continuation checkpoint found under $experimentRoot" }

$loadRun = $checkpoint.Directory.Name
$checkpointName = $checkpoint.Name
$name = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $backgroundDir "$name.log"
$exitCodeFile = Join-Path $backgroundDir "$name.exitcode"
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'E:\NVIDIA\Run-G1-Smart-Rough-Worker.ps1',
    '-NumEnvs', $NumEnvs, '-MaxIterations', $MaxIterations,
    '-LoadRun', $loadRun, '-Checkpoint', $checkpointName,
    '-LogFile', $log, '-ExitCodeFile', $exitCodeFile
)
$process = Start-Process powershell.exe -ArgumentList $arguments -WindowStyle Hidden -PassThru
Set-Content (Join-Path $backgroundDir 'pid.txt') $process.Id
Set-Content (Join-Path $backgroundDir 'latest.txt') $name
Write-Output 'Smart rough-terrain training with edge return started in the background.'
Write-Output "PID: $($process.Id)"
Write-Output "Robots: $NumEnvs"
Write-Output "Resuming: $loadRun\$checkpointName"
Write-Output "Log: $log"
