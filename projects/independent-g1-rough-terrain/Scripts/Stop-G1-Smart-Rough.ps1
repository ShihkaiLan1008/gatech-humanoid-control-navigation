$p=Get-CimInstance Win32_Process|Where-Object{$_.CommandLine -like '*Isaac-Energy-Smart-Rough-G1*'}
if(-not$p){Write-Output 'Smart rough training is not running.';exit}
$root=$p|Where-Object{$p.ProcessId -notcontains $_.ParentProcessId}|Select-Object -First 1;if(-not$root){$root=$p|Select-Object -First 1}
& taskkill.exe /PID $root.ProcessId /T /F|Out-Null;Write-Output "Stopped process tree $($root.ProcessId)."
