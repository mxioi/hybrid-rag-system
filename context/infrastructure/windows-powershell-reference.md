# Windows PowerShell & RSAT Reference for Local LLM

This document provides PowerShell commands and patterns for managing Windows Server infrastructure via SSH. DC1 (<ADD-IP-ADDRESS>) has RSAT modules installed and can manage remote servers.

## SSH Access Pattern

```bash
# From Unraid, SSH to Windows servers:
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> \
    svc-monitor@<ADD-IP-ADDRESS> "powershell -Command 'Your-Command'"

# Or use the SSH config alias (if configured):
ssh DC1 "powershell -Command 'Get-Service'"
```

## Server Inventory

| Alias | IP | Role | RSAT Available |
|-------|-----|------|----------------|
| DC1 | <ADD-IP-ADDRESS> | Primary DC | Yes (all modules) |
| DC2 | <ADD-IP-ADDRESS> | Secondary DC | Partial |
| APP1 | 10.0.0.X | SCCM/MECM | No |
| EXC1 | 10.0.0.X | Exchange | Exchange tools |
| FS1 | 10.0.0.X | File Server | No |
| Entra | 10.0.0.X | Entra ID Connect | AAD module |

---

## Active Directory Commands (Run from DC1)

### Health Checks

```powershell
# Check AD replication status
repadmin /replsummary

# Detailed replication status
repadmin /showrepl

# Force replication
repadmin /syncall /A /e

# DC diagnostics (quiet mode - errors only)
dcdiag /q

# Full DC diagnostics
dcdiag /v

# Check FSMO role holders
netdom query fsmo

# List all domain controllers
Get-ADDomainController -Filter * | Select Name, IPv4Address, Site, IsGlobalCatalog, OperatingSystem
```

### User Management

```powershell
# Find locked out accounts
Search-ADAccount -LockedOut | Select Name, SamAccountName, LockedOut

# Unlock a user
Unlock-ADAccount -Identity "username"

# Get user details
Get-ADUser -Identity "username" -Properties *

# Find users by name pattern
Get-ADUser -Filter "Name -like '*pattern*'" | Select Name, SamAccountName, Enabled

# List recently created users (last 7 days)
$date = (Get-Date).AddDays(-7)
Get-ADUser -Filter {Created -gt $date} -Properties Created | Select Name, Created

# Disable a user account
Disable-ADAccount -Identity "username"

# Reset password (generates random)
Set-ADAccountPassword -Identity "username" -Reset -NewPassword (ConvertTo-SecureString "TempP@ss123!" -AsPlainText -Force)
```

### Computer Management

```powershell
# List all computers
Get-ADComputer -Filter * | Select Name, DNSHostName, Enabled

# Find inactive computers (90 days)
$date = (Get-Date).AddDays(-90)
Get-ADComputer -Filter {LastLogonDate -lt $date} -Properties LastLogonDate | Select Name, LastLogonDate

# Get computer details
Get-ADComputer -Identity "computername" -Properties *
```

### Group Management

```powershell
# List group members
Get-ADGroupMember -Identity "GroupName" | Select Name, SamAccountName

# Add user to group
Add-ADGroupMember -Identity "GroupName" -Members "username"

# Remove user from group
Remove-ADGroupMember -Identity "GroupName" -Members "username" -Confirm:$false
```

---

## DNS Management (Run from DC1 or DC2)

```powershell
# List DNS zones
Get-DnsServerZone

# Get DNS records for a zone
Get-DnsServerResourceRecord -ZoneName "homelab.local"

# Add A record
Add-DnsServerResourceRecordA -ZoneName "homelab.local" -Name "newhost" -IPv4Address "10.0.0.X"

# Remove DNS record
Remove-DnsServerResourceRecord -ZoneName "homelab.local" -Name "oldhost" -RRType A -Force

# Clear DNS cache
Clear-DnsServerCache

# Test DNS resolution
Resolve-DnsName -Name "dc1.homelab.local" -Server <ADD-IP-ADDRESS>
```

---

## DHCP Management (Run from DC1)

```powershell
# List DHCP scopes
Get-DhcpServerv4Scope

# Get scope statistics
Get-DhcpServerv4ScopeStatistics

# List active leases
Get-DhcpServerv4Lease -ScopeId "10.0.0.X"

# Add reservation
Add-DhcpServerv4Reservation -ScopeId "10.0.0.X" -IPAddress "<ADD-IP-ADDRESS>" -ClientId "AA-BB-CC-DD-EE-FF" -Name "device-name"

# Remove reservation
Remove-DhcpServerv4Reservation -ScopeId "10.0.0.X" -IPAddress "<ADD-IP-ADDRESS>"
```

---

## Group Policy Management (Run from DC1)

```powershell
# List all GPOs
Get-GPO -All | Select DisplayName, GpoStatus, CreationTime

# Get GPO details
Get-GPO -Name "GPO Name" | Get-GPOReport -ReportType HTML -Path "C:\temp\gpo-report.html"

# Force GP update on remote computer
Invoke-GPUpdate -Computer "computername" -Force

# Get GPO links
Get-ADOrganizationalUnit -Filter * | ForEach-Object {
    $ou = $_
    (Get-GPInheritance -Target $ou.DistinguishedName).GpoLinks |
    Select @{N='OU';E={$ou.Name}}, DisplayName
}
```

---

## SCCM/MECM Commands (Run from APP1)

### Service Health

```powershell
# Check all SCCM services
Get-Service SMS_* | Select Name, Status, StartType

# Check WSUS service
Get-Service WsusService | Select Name, Status

# Check SQL Server
Get-Service MSSQLSERVER | Select Name, Status

# Restart SCCM Executive service
Restart-Service SMS_EXECUTIVE

# Check IIS
Get-Service W3SVC | Select Name, Status
```

### SCCM Client Operations

```powershell
# Trigger machine policy on remote client
Invoke-WmiMethod -Namespace root\ccm -Class SMS_Client -Name TriggerSchedule -ArgumentList "{00000000-0000-0000-0000-000000000021}"

# Check client cache size
Get-WmiObject -Namespace root\ccm\SoftMgmtAgent -Class CacheConfig | Select Size

# Get SCCM client version on remote machine
Get-WmiObject -Namespace root\ccm -Class SMS_Client -ComputerName "targetPC" | Select ClientVersion
```

### Content Distribution

```powershell
# Check DP status (run on SCCM server)
Get-CMDistributionStatus | Select PackageID, Targeted, NumberSuccess, NumberErrors

# Validate content on DP
Invoke-CMContentValidation -DistributionPointName "dp.homelab.local"
```

---

## Exchange Commands (Run from EXC1)

```powershell
# Connect to Exchange (if remote)
Add-PSSnapin Microsoft.Exchange.Management.PowerShell.SnapIn

# List mailboxes
Get-Mailbox | Select Name, PrimarySmtpAddress, Database

# Get mailbox statistics
Get-MailboxStatistics -Identity "user@homelab.local" | Select DisplayName, TotalItemSize, ItemCount

# Check mail queues
Get-Queue | Select Identity, Status, MessageCount

# Get database status
Get-MailboxDatabase -Status | Select Name, Mounted, DatabaseSize

# Test mail flow
Send-MailMessage -From "test@homelab.local" -To "admin@homelab.local" -Subject "Test" -Body "Test email" -SmtpServer "localhost"
```

---

## Entra ID Connect Commands (Run from Entra server)

```powershell
# Check sync status
Get-ADSyncScheduler

# Force delta sync
Start-ADSyncSyncCycle -PolicyType Delta

# Force full sync (use sparingly)
Start-ADSyncSyncCycle -PolicyType Initial

# Check connector status
Get-ADSyncConnector | Select Name, Type

# Get last sync time
Get-ADSyncScheduler | Select NextSyncCycleStartTimeInUTC, LastSyncCycleStartTimeInUTC
```

---

## General Windows Server Commands

### Service Management

```powershell
# List all services
Get-Service | Select Name, Status, StartType

# Check specific service
Get-Service -Name "ServiceName" | Select *

# Start/Stop/Restart
Start-Service -Name "ServiceName"
Stop-Service -Name "ServiceName"
Restart-Service -Name "ServiceName"

# Set service to auto-start
Set-Service -Name "ServiceName" -StartupType Automatic
```

### Disk & Storage

```powershell
# Check disk space
Get-PSDrive -PSProvider FileSystem | Select Name, @{N='Used(GB)';E={[math]::Round($_.Used/1GB,2)}}, @{N='Free(GB)';E={[math]::Round($_.Free/1GB,2)}}

# Detailed disk info
Get-WmiObject Win32_LogicalDisk | Select DeviceID, @{N='Size(GB)';E={[math]::Round($_.Size/1GB,2)}}, @{N='Free(GB)';E={[math]::Round($_.FreeSpace/1GB,2)}}
```

### Event Logs

```powershell
# Get recent errors (last 24h)
Get-EventLog -LogName System -EntryType Error -After (Get-Date).AddDays(-1) | Select TimeGenerated, Source, Message | Format-Table -Wrap

# Get security events (failed logins)
Get-EventLog -LogName Security -InstanceId 4625 -After (Get-Date).AddDays(-1) | Select TimeGenerated, Message

# Get application errors
Get-EventLog -LogName Application -EntryType Error -After (Get-Date).AddDays(-1) | Select TimeGenerated, Source, Message
```

### System Info

```powershell
# Get system uptime
(Get-CimInstance Win32_OperatingSystem).LastBootUpTime

# Get installed hotfixes
Get-HotFix | Select HotFixID, InstalledOn | Sort InstalledOn -Descending | Select -First 10

# Get pending reboot status
Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
```

### Network

```powershell
# Get IP configuration
Get-NetIPConfiguration

# Get DNS settings
Get-DnsClientServerAddress | Select InterfaceAlias, ServerAddresses

# Test connectivity
Test-NetConnection -ComputerName "<ADD-IP-ADDRESS>" -Port 445

# Get active connections
Get-NetTCPConnection -State Established | Select LocalAddress, LocalPort, RemoteAddress, RemotePort
```

---

## Remote Execution Patterns

### Run command on remote Windows server from Unraid

```bash
# Single command
ssh svc-monitor@<ADD-IP-ADDRESS> "powershell -Command 'Get-Service | Where Status -eq Running'"

# Multi-line script
ssh svc-monitor@<ADD-IP-ADDRESS> "powershell -Command '
Get-Service SMS_* | ForEach-Object {
    Write-Output \"$($_.Name): $($_.Status)\"
}
'"
```

### Use DC1 as jump host for RSAT commands

```bash
# Manage remote server's AD from DC1
ssh DC1 "powershell -Command 'Get-ADComputer -Server DC2 -Filter *'"

# Invoke command on remote Windows server from DC1
ssh DC1 "powershell -Command 'Invoke-Command -ComputerName APP1 -ScriptBlock { Get-Service SMS_* }'"
```

---

## Troubleshooting Patterns

### AD Replication Issues

```powershell
# 1. Check replication status
repadmin /replsummary

# 2. Check for replication failures
repadmin /showrepl * /csv | ConvertFrom-Csv | Where {$_.'Number of Failures' -gt 0}

# 3. Force sync if needed
repadmin /syncall DC1 /A /e /P

# 4. Check AD database
esentutl /g "C:\Windows\NTDS\ntds.dit"
```

### SCCM Client Not Reporting

```powershell
# 1. Check client health
Get-WmiObject -Namespace root\ccm -Class SMS_Client | Select ClientVersion

# 2. Repair client
Start-Process "C:\Windows\CCM\ccmrepair.exe"

# 3. Force policy refresh
Invoke-WmiMethod -Namespace root\ccm -Class SMS_Client -Name TriggerSchedule -ArgumentList "{00000000-0000-0000-0000-000000000021}"
```

### DNS Resolution Failures

```powershell
# 1. Clear DNS cache
Clear-DnsClientCache

# 2. Test resolution
Resolve-DnsName "hostname.homelab.local"

# 3. Check DNS server
nslookup hostname.homelab.local <ADD-IP-ADDRESS>

# 4. Register DNS
ipconfig /registerdns
```
