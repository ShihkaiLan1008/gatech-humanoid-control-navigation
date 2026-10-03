param(
    [int]$NumEnvs = 500,
    [int]$MaxIterations = 2500,
    [Parameter(Mandatory=$true)][string]$LoadRun,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$LogFile,
    [Parameter(Mandatory=$true)][string]$ExitCodeFile
)

$ErrorActionPreference = 'Stop'
$env:VIRTUAL_ENV = 'E:\NVIDIA\env_isaaclab'
$env:Path = 'E:\NVIDIA\env_isaaclab\Scripts;E:\NVIDIA\portable-git\cmd;E:\NVIDIA\tools;' + $env:Path
$env:UV_CACHE_DIR = 'E:\NVIDIA\uv-cache'
$env:GIT_PYTHON_GIT_EXECUTABLE = 'E:\NVIDIA\portable-git\cmd\git.exe'
Set-Location -LiteralPath 'E:\NVIDIA\IsaacLab'

try {
    "[$(Get-Date -Format o)] Resuming smart rough G1 training with edge return: robots=$NumEnvs iterations=$MaxIterations run=$LoadRun checkpoint=$Checkpoint" | Add-Content $LogFile
    $command = '"E:\NVIDIA\IsaacLab\isaaclab.bat" train --rl_library rsl_rl ' +
        '--task=Isaac-Energy-Smart-Rough-G1 ' +
        "--num_envs=$NumEnvs --max_iterations=$MaxIterations --seed=42 " +
        "--resume --load_run=$LoadRun --checkpoint=$Checkpoint " +
        'physics=newton_mjwarp >> "' + $LogFile + '" 2>&1'
    & cmd.exe /d /s /c $command
    $code = $LASTEXITCODE
} catch {
    $_ | Out-String | Add-Content $LogFile
    $code = 1
} finally {
    "[$(Get-Date -Format o)] Smart rough training exited with code $code" | Add-Content $LogFile
    Set-Content $ExitCodeFile $code
}
exit $code

