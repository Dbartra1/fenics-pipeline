# scripts/setup_wsl.ps1
# Run from Windows PowerShell (not WSL2): .\scripts\setup_wsl.ps1
# Requires PowerShell 5.1+ — no admin rights needed for .wslconfig

param(
    [int]$MemoryGB   = 24,
    [int]$Processors = 10,
    [int]$SwapGB     = 8
)

$wslconfig = "$env:USERPROFILE\.wslconfig"
$backup    = "$wslconfig.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Back up existing config if present
if (Test-Path $wslconfig) {
    Copy-Item $wslconfig $backup
    Write-Host "Backed up existing .wslconfig to $backup"
}

# Detect total physical RAM — cap memory recommendation at 80% of physical
$physicalRAM = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$recommended = [math]::Floor($physicalRAM * 0.8)
if ($MemoryGB -gt $recommended) {
    Write-Warning "Requested ${MemoryGB}GB exceeds 80% of physical RAM (${physicalRAM}GB). Capping at ${recommended}GB."
    $MemoryGB = $recommended
}

$content = @"
[wsl2]
memory=${MemoryGB}GB
processors=${Processors}
swap=${SwapGB}GB
"@

Set-Content -Path $wslconfig -Value $content -Encoding UTF8
Write-Host "Written to $wslconfig"
Write-Host ""
Write-Host "Contents:"
Get-Content $wslconfig

# Shut down WSL2 so the new config takes effect
Write-Host ""
Write-Host "Shutting down WSL2 to apply config..."
wsl --shutdown
Start-Sleep -Seconds 3

# Verify
$result = wsl -- bash -c "free -h | awk '/^Mem:/{print \$2}'"
Write-Host "WSL2 memory now reported as: $result"
Write-Host ""
Write-Host "Done. Start your WSL2 terminal and run: make build && make up"