param(
    [ValidateRange(1,32)]
    [int]$NumRobots = 8
)

$ErrorActionPreference = 'Stop'
$experimentRoot = 'E:\NVIDIA\IsaacLab\logs\rsl_rl\g1_energy_smart_rough'
$checkpoint = Get-ChildItem -LiteralPath $experimentRoot -Recurse -Filter 'model_*.pt' |
    Where-Object { $_.Directory.Name -ne 'seed_from_energy_medium' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $checkpoint) {
    throw "No trained smart-rough checkpoint was found under $experimentRoot"
}
$checkpointPath = $checkpoint.FullName

$env:VIRTUAL_ENV = 'E:\NVIDIA\env_isaaclab'
$env:Path = 'E:\NVIDIA\env_isaaclab\Scripts;E:\NVIDIA\portable-git\cmd;E:\NVIDIA\tools;' + $env:Path
$env:UV_CACHE_DIR = 'E:\NVIDIA\uv-cache'
$env:GIT_PYTHON_GIT_EXECUTABLE = 'E:\NVIDIA\portable-git\cmd\git.exe'
Set-Location -LiteralPath 'E:\NVIDIA\IsaacLab'

Write-Host "Opening $NumRobots terrain-aware G1 robots."
Write-Host "Checkpoint: $checkpointPath"
Write-Host 'The background trainer will continue independently. Close the viewer to stop playback.'

& 'E:\NVIDIA\IsaacLab\isaaclab.bat' play `
    --rl_library rsl_rl `
    --task=Isaac-Energy-Smart-Rough-G1 `
    --num_envs=$NumRobots `
    --checkpoint=$checkpointPath `
    --visualizer newton `
    --max_visible_envs=$NumRobots `
    --real-time `
    physics=newton_mjwarp

if ($LASTEXITCODE -ne 0) {
    Read-Host "Viewer exited with code $LASTEXITCODE. Press Enter to close"
}
