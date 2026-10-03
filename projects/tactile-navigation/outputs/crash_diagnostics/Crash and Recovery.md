# Windows crash and research recovery

3 October 2026, Asia/Taipei.

## Confirmed evidence

Windows boot time: 16:33:46. The System log reports a bugcheck at 16:34:08:

```text
0x00000139
parameters: 0x3, 0xffffc8051200ecf0, 0xffffc8051200ec48, 0x0
dump: C:\Windows\Minidump\100326-22312-01.dmp
```

Event 1001 records the bugcheck; event 41 records the unclean reboot. Microsoft documents 0x139 as KERNEL_SECURITY_CHECK_FAILURE and parameter 1 = 3 as LIST_ENTRY corruption. This confirms a Windows kernel crash, not a particular defective driver. Corruption may be detected later than it occurs.

Source: https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x139--kernel-security-check-failure

The minidump is present but reading/copying it was denied by Windows permissions. No Windows debugger is available in the checked locations. The Microsoft debugger download descriptor was retrieved, but no debugger was installed or run. A dump analysis is required to narrow down the driver/component; no permanent root-cause repair is claimed.

The WER kernel report folder is listed but its contents were also inaccessible. Unrelated post-boot domain/network warnings are not evidence for the cause of this crash. I did not change corporate security software, device drivers, BIOS settings, or Windows access controls.

## Research recovery

- Original `room2`: completed; physical goal reached, framework `arrival_uncertain`, no fall. Evaluation, replay states, and finished video preserved.
- New `sky_island_choice`: completed; physical goal reached, framework `arrival_uncertain`, no fall. Evaluation, replay states, and finished video preserved.
- New `sky_offset_gate`: system reboot interrupted the original attempt at about 133 seconds of simulation. Its journal is preserved and recovered as `recovered.eval.json` with outcome `system_restart`; it is separate from completed trials.
- New `sky_staggered_baffles`: had not started.
- Offset gate and staggered baffles restarted in a fresh `out/sky_evaluation_20261003/new_maps_physics_restart/` folder. Earlier artifacts were not overwritten.

## Precautions on restart

Only one physical simulation process runs at a time. Video rendering waits until physical simulations finish. CPU allocation is reduced to two cores, and the existing runner monitors a 4 GiB process-group RSS limit. Seed, maps, controller checkpoint, navigation settings, and 900-second simulation cap remain unchanged.

The first post-restart check showed CUDA available, about 1.1 GiB GPU memory in use, GPU temperature 58 C, and about 9.9 GiB free Windows physical memory. The installed NVIDIA driver reports 610.74. These are snapshots after restart, not measurements at the crash and not proof of its cause. Reduced concurrency is a precaution, not a verified fix for kernel corruption.

## If exact driver diagnosis is needed

An administrator can copy `C:\Windows\Minidump\100326-22312-01.dmp` into this `outputs/crash_diagnostics/` folder. Analyze that accessible copy with Microsoft's debugger locally. Keep dump files private: they can contain memory from the machine. Do not publish them in the college-application repository.
