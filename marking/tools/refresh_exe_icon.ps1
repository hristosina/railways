param(
    [Parameter(Mandatory = $true)]
    [string]$ExecutablePath
)

$resolvedPath = (Resolve-Path -LiteralPath $ExecutablePath).Path

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class MarkingShellNotify
{
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    public static extern void SHChangeNotify(
        uint eventId,
        uint flags,
        string item1,
        IntPtr item2
    );
}
"@

# SHCNE_UPDATEITEM + SHCNF_PATHW: invalidate the cached shell data for this EXE.
[MarkingShellNotify]::SHChangeNotify(0x00002000, 0x0005, $resolvedPath, [IntPtr]::Zero)

# SHCNE_ASSOCCHANGED: make open Explorer windows redraw their icon lists.
[MarkingShellNotify]::SHChangeNotify(0x08000000, 0x0000, $null, [IntPtr]::Zero)

Write-Host "Windows Shell icon refreshed for $resolvedPath"
