$dir='E:\NVIDIA\smart-rough-background';$latest=Join-Path $dir 'latest.txt';if(-not(Test-Path $latest)){Write-Output 'No smart rough job exists.';exit}
$name=(Get-Content $latest -Raw).Trim();$log=Join-Path $dir "$name.log";$exit=Join-Path $dir "$name.exitcode"
$p=Get-CimInstance Win32_Process|Where-Object{$_.CommandLine -like '*Isaac-Energy-Smart-Rough-G1*'}
if($p){Write-Output 'Status: RUNNING';nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader}elseif(Test-Path $exit){Write-Output "Status: FINISHED (exit code $((Get-Content $exit -Raw).Trim()))"}else{Write-Output 'Status: NOT RUNNING'}
Write-Output "Log: $log";if(Test-Path $log){Get-Content $log -Tail 45}

