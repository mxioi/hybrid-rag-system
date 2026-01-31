# Infrastructure Automation Guide

## Overview

This guide provides comprehensive prompt templates and automation workflows for managing your homelab infrastructure through Claude Code. These templates leverage the infrastructure context documents to enable intelligent, autonomous monitoring and management.

## Quick Reference

### Available Infrastructure
- **Unraid Server** (<ADD-IP-ADDRESS>) - NAS/Hypervisor host
- **SCCM Server** (10.0.0.X) - VM3-APP1, Systems management
- **Domain Controller 1** (<ADD-IP-ADDRESS>) - VM1-DC1, Active Directory
- **Domain Controller 2** (10.0.0.X) - VM5-DC2, Secondary DC 
- **Proxmox Server** (10.0.0.X) - VM host
- **UDM Pro Max** (<ADD-IP-ADDRESS>) - Network gateway

### Monitoring Scripts
- `/path/to/hybrid-rag-system/context/tools/check-unraid.sh` (TBD)
- `/path/to/hybrid-rag-system/context/tools/check-sccm.sh`
- `/path/to/hybrid-rag-system/context/tools/check-proxmox.sh`
- `/path/to/hybrid-rag-system/context/tools/check-udm.sh`

### Context Documents
- `/path/to/hybrid-rag-system/context/infrastructure/unraid-server.md`
- `/path/to/hybrid-rag-system/context/infrastructure/sccm-server.md`
- `/path/to/hybrid-rag-system/context/infrastructure/domain-controller.md`
- `/path/to/hybrid-rag-system/context/infrastructure/udm-pro-max.md`

## Discord Bot Interaction Prompts

### General Status Checks

#### Full Infrastructure Health Check
```
Check the health of my entire homelab infrastructure and provide a summary:
1. Unraid server - array status, docker containers, disk space
2. SCCM server - services, disk space, recent errors
3. Domain Controller - AD services, replication status
4. Proxmox - system resources, VM status
5. Network (UDM) - WAN status, access points, client count

Format as executive dashboard with ✅/⚠️/❌ status for each component.
```

#### Quick Status Check
```
Give me a quick status update on all infrastructure:
- Unraid array and key containers
- SCCM and DC services
- Network connectivity and client count
Format as brief bullet list.
```

### Unraid Server Prompts

#### Daily Unraid Check
```
Check my Unraid server status:
1. Array status (all disks online?)
2. Critical containers running (lldap, netdata, plex, claude-code)
3. Disk space on array and cache
4. Docker image disk usage
5. Recent CrowdSec alerts

Report with status indicators and highlight any issues.
```

#### Container Management
```
Check the status of [CONTAINER_NAME] container on Unraid:
1. Is it running?
2. Recent log entries (last 50 lines)
3. Resource usage (CPU/memory)
4. Health check status
If unhealthy, provide troubleshooting steps.
```

#### Disk Space Management
```
My Unraid server is running low on disk space. Please:
1. Check current usage on array and cache
2. Identify top 10 largest directories in /mnt/user
3. Check for old docker logs (>100MB)
4. Look for old downloads or media files
5. Provide cleanup recommendations

DO NOT auto-delete anything - just provide recommendations with size estimates.
```

### SCCM Server Prompts

#### SCCM Daily Health
```
Check SCCM server (VM3-APP1) health:
1. All SMS_* services running?
2. Disk space on C: drive
3. Any critical errors in last 24 hours
4. WSUS last sync status
5. Active client count

Format as health report with status indicators.
```

#### Deployment Status Check
```
Check the status of SCCM deployment "[DEPLOYMENT_NAME]":
1. How many clients are targeted?
2. Success/failed/in-progress counts
3. Any common error codes?
4. Content distribution status
Provide deployment summary with success percentage.
```

#### Client Health Check
```
Why is computer [COMPUTER_NAME] not reporting to SCCM?
1. Check if computer exists in SCCM database
2. Last heartbeat discovery time
3. Recent status messages for this client
4. Check for deployment errors
5. Provide remediation steps
```

#### Patch Compliance
```
Generate a patch compliance report for SCCM:
1. WSUS last sync time and status
2. Number of pending critical updates
3. Number of pending security updates
4. Top 5 missing updates across all clients
5. Overall compliance percentage

Format as executive summary.
```

### Domain Controller Prompts

#### AD Daily Health
```
Check Active Directory health on VM1-DC1:
1. NTDS and DNS services running?
2. AD replication status (expect DC2 failure)
3. Any authentication failures (last 24 hours)
4. Disk space on C: drive
5. Time synchronization status

Report with status indicators.
```

#### User Account Management
```
Perform AD user account audit:
1. Total user count
2. Newly created accounts (last 7 days)
3. Currently disabled accounts
4. Locked out accounts
5. Accounts with passwords expiring soon (next 7 days)

Format as summary with lists of specific accounts.
```

#### Replication Troubleshooting
```
Check AD replication status between DC1 and DC2:
1. Run repadmin /replsummary check
2. Identify specific replication errors
3. Check time sync between DCs
4. Verify DNS resolution for both DCs
5. Test RPC connectivity

Provide troubleshooting report with recommendations to fix DC2 replication.
```

#### Security Audit
```
Run security audit on Domain Controller:
1. Failed login attempts (last 24 hours) - Event ID 4625
2. Account lockouts (last 7 days) - Event ID 4740
3. Unusual authentication patterns
4. Recent privileged account usage (Domain Admins)

Format as security report with threat assessment.
```

#### Group Policy Check
```
Check Group Policy health in Active Directory:
1. List all GPOs with status
2. Verify SYSVOL replication for each GPO
3. Identify any unlinked GPOs
4. Find empty GPOs (no settings configured)
5. Recent GPO apply errors

Provide GPO health summary with recommendations.
```

### Network (UDM) Prompts

#### Network Health Check
```
Check my UDM Pro Max network status:
1. WAN connectivity (both interfaces if dual-WAN)
2. Internet status
3. Access point health and client counts
4. Total connected clients
5. Any active network alerts

Format as network dashboard with status indicators.
```

#### Bandwidth Analysis
```
Analyze bandwidth usage on my network:
1. Top 5 bandwidth consumers (by total bytes)
2. WAN utilization percentage
3. Any clients with unusual traffic patterns
4. Average bandwidth per client

Provide bandwidth report with recommendations if anyone hogging bandwidth.
```

#### WiFi Troubleshooting
```
I'm having WiFi issues. Please check:
1. All access points online?
2. Client distribution across APs (balanced?)
3. Any clients with connection issues
4. Channel interference or congestion
5. Recent AP disconnections

Provide WiFi troubleshooting report with recommendations.
```

#### Client Connectivity Issue
```
Client [HOSTNAME/IP] cannot connect to network:
1. Is client visible in UniFi controller?
2. Which AP is client trying to connect to?
3. Any DHCP issues?
4. Check for firewall blocks
5. Verify VLAN assignment

Provide troubleshooting steps to restore connectivity.
```

### Cross-System Prompts

#### Complete Infrastructure Report
```
Generate comprehensive infrastructure report:

**Unraid Server**:
- Array status and disk health
- Top resource-consuming containers
- Disk space usage trends

**SCCM/Configuration Management**:
- Client health percentage
- Recent deployment success rates
- Pending patches summary

**Active Directory**:
- Total users/computers
- Replication health
- Recent security events

**Network**:
- WAN uptime and bandwidth
- Client count trends
- WiFi performance

Format as executive summary with key metrics and alerts.
```

#### Security Posture Review
```
Review security posture across all infrastructure:
1. Unraid: CrowdSec alerts and failed login attempts
2. SCCM: Pending security updates
3. AD: Failed authentications and lockouts
4. Network: IPS/IDS alerts and suspicious traffic

Provide overall security assessment with threat level (Low/Medium/High).
```

#### Disaster Recovery Readiness
```
Check disaster recovery readiness:
1. Unraid: Last successful backup (duplicati)
2. SCCM: Site backup status
3. AD: System state backup status
4. Network: UDM configuration backup
5. Document any systems without recent backups

Provide DR readiness report with recommendations.
```

## n8n Workflow Templates

### Scheduled Monitoring Workflows

#### Workflow 1: Hourly Infrastructure Health Check
```javascript
// n8n workflow - Every hour
// Nodes: Schedule, SSH (Claude), Discord Webhook

Schedule Trigger: Cron 0 * * * *

SSH to Unraid (Execute Command):
claude ask "Quick infrastructure health check - just critical services status for Unraid, SCCM, DC, and network. Brief format."

Parse Response:
Filter for ⚠️ or ❌ indicators

Discord Webhook (if issues found):
Send alert to #infrastructure-alerts channel
```

#### Workflow 2: Daily Summary Report
```javascript
// n8n workflow - Daily at 8 AM
// Nodes: Schedule, SSH (Claude), Email/Discord

Schedule Trigger: Cron 0 8 * * *

SSH to Unraid (Execute Command):
claude ask "Generate daily infrastructure summary including:
- Key metrics from all systems
- Any issues encountered yesterday
- Disk space status
- Backup status
- Top 5 bandwidth users
Format as daily digest email."

Email Node:
Send to: admin@homelab.local
Subject: Daily Infrastructure Report - {date}
Body: {claude_response}
```

#### Workflow 3: Real-time Container Monitoring
```javascript
// n8n workflow - Every 5 minutes
// Nodes: Schedule, HTTP Request, If, Discord

Schedule Trigger: Every 5 minutes

Check Critical Containers:
For each: lldap, netdata, claude-code, plex
  HTTP: Check health endpoint OR docker ps query

If Container Down:
  Discord Alert: "@admin Container {name} is down!"
  SSH to Unraid: docker restart {container}
  Wait 30 seconds
  Verify restart successful
  Discord Update: Status of restart attempt
```

#### Workflow 4: Bandwidth Alert
```javascript
// n8n workflow - Every 15 minutes
// Nodes: Schedule, SSH, Function, Discord

Schedule Trigger: Every 15 minutes

Query UDM API:
curl network stats endpoint

Function Node:
Calculate total bandwidth used in last 15min
Check if > threshold (e.g., 50GB/hour rate)

If Threshold Exceeded:
  Claude Analysis:
  "My network bandwidth is very high. Check:
   1. Top bandwidth consumers
   2. Any unusual traffic patterns
   3. Possible explanations
   Provide quick analysis."

  Discord Alert:
  Send analysis to #network-alerts
```

#### Workflow 5: Security Event Aggregator
```javascript
// n8n workflow - Every 30 minutes
// Nodes: Schedule, SSH (multiple), Merge, If, Discord

Schedule Trigger: Every 30 minutes

Parallel SSH Queries:
1. Unraid: Check CrowdSec alerts
2. SCCM: Check failed deployments
3. DC: Check failed auth attempts (Event 4625)
4. UDM: Check IPS/IDS alerts

Merge Results

If Security Events Found:
  Claude Analysis:
  "Security events detected:
  {event_summary}

  Analyze severity and provide recommendations."

  Discord Alert:
  "#security-alerts {claude_response}"
```

### Reactive Workflows (Webhook Triggered)

#### Workflow 6: Service Down Auto-Recovery
```javascript
// n8n workflow - Webhook trigger
// Triggered by monitoring system when service fails

Webhook Trigger: POST /webhook/service-down
Body: {service: "name", system: "hostname"}

Claude Troubleshooting:
"Service {service} is down on {system}. Please:
1. Check service status
2. Review recent logs for errors
3. Attempt restart if safe
4. Verify dependencies
Report actions taken and success/failure."

Discord Notification:
"Service Recovery Attempt: {service} on {system}
{claude_response}"

If Recovery Failed:
  Escalate: Page administrator
```

#### Workflow 7: Disk Space Emergency
```javascript
// n8n workflow - Webhook trigger
// Triggered when disk space < 5% free

Webhook Trigger: POST /webhook/disk-alert
Body: {system: "hostname", disk: "path", percent: 95}

Claude Emergency Cleanup:
"URGENT: Disk {disk} on {system} is {percent}% full.
Safely clean up:
1. Old log files (>30 days)
2. Temp files
3. Docker unused images
4. Identify large files for manual review
Execute safe automated cleanup ONLY."

Monitor for 15 minutes:
  Check if space freed

Discord Update:
  "Disk Space Alert - {system}
   Before: {percent}%
   After: {new_percent}%
   {claude_actions}"
```

## Advanced Automation Scenarios

### Scenario 1: Proactive Patch Management
```
n8n Workflow (Weekly - Sunday 2 AM):
1. Claude: Check WSUS for new critical patches
2. If critical patches available:
   a. Claude: Create SCCM test deployment to pilot collection
   b. Wait 24 hours
   c. Claude: Check pilot deployment success rate
   d. If >95% success:
      - Claude: Deploy to production collections
      - Discord: Notify of patch rollout
   e. If <95% success:
      - Discord: Alert admin, hold production deployment
3. Send weekly patch status report
```

### Scenario 2: Intelligent Container Health Management
```
n8n Workflow (Every 10 minutes):
1. Check health status of all critical containers
2. For any unhealthy container:
   a. Claude: Analyze last 100 log lines
   b. Claude: Determine if transient or persistent issue
   c. If transient: Wait and monitor
   d. If persistent:
      - Attempt restart
      - Verify dependencies (network, storage)
      - Check resource limits
      - If still failing: Alert admin with diagnosis
3. Track restart history, alert if same container fails >3 times/day
```

### Scenario 3: Network Performance Optimizer
```
n8n Workflow (Daily - off-peak hours):
1. Claude: Analyze 24h bandwidth patterns
2. Claude: Identify bandwidth trends and anomalies
3. Claude: Check WiFi channel utilization
4. If optimization opportunities found:
   a. Generate recommendations
   b. Request admin approval for changes
   c. If approved: Apply optimizations
   d. Monitor for 1 hour
   e. Rollback if performance degrades
5. Send daily network performance report
```

### Scenario 4: Automated Backup Verification
```
n8n Workflow (Daily - 3 AM):
1. Claude: Check all backup systems:
   - Unraid duplicati status
   - SCCM site backup
   - VM snapshots on Proxmox
   - UDM configuration backup
2. For each backup:
   a. Verify completion timestamp (<24h)
   b. Check backup size (not zero, within expected range)
   c. Verify backup integrity if possible
3. If any backup failed:
   a. Attempt re-trigger
   b. Alert admin if retry fails
4. Generate backup health report
```

### Scenario 5: Holistic Security Monitoring
```
n8n Workflow (Every 30 minutes):
1. Claude: Aggregate security events from all systems
2. Claude: Correlate events across systems
   Example: Failed AD auth + CrowdSec alert from same IP
3. Claude: Risk assessment and threat classification
4. If Medium/High threat:
   a. Gather additional context
   b. Check if IP in threat intel feeds
   c. Verify if legitimate user/system
   d. Recommend blocking/quarantine
   e. Alert admin with full context
5. Daily security digest sent to admin
```

## Prompt Engineering Best Practices

### Clear and Specific
❌ Bad: "Check the servers"
✅ Good: "Check SCCM server services, disk space, and recent errors"

### Structured Requests
❌ Bad: "Look at everything"
✅ Good: "Check: 1) Service status 2) Disk space 3) Recent logs 4) Performance metrics"

### Set Expectations
❌ Bad: "Fix the issue"
✅ Good: "Diagnose the issue and provide recommendations. If safe, auto-fix. If risky, provide manual steps."

### Provide Context
❌ Bad: "Container won't start"
✅ Good: "The lldap container on Unraid won't start. It was working yesterday. Check logs, volumes, and port conflicts."

### Define Output Format
❌ Bad: "Give me a report"
✅ Good: "Format as markdown table with columns: System, Status, Issues, Action Required"

### Set Scope
❌ Bad: "Optimize everything"
✅ Good: "Check docker container resource usage. If any container using >80% CPU, investigate and provide tuning recommendations."

## Emergency Response Prompts

### Critical Service Down
```
URGENT: {SERVICE} on {SYSTEM} is down and affecting users.

Immediate actions:
1. Verify service is actually down (not false alarm)
2. Check dependencies (network, database, storage)
3. Review last 10 minutes of logs before failure
4. Attempt service restart if no data risk
5. Check for resource exhaustion (disk, memory, CPU)
6. Provide immediate workaround if restart fails

Execute quickly and report status every 2 minutes until resolved.
```

### Security Breach Suspected
```
SECURITY ALERT: Suspicious activity detected from IP {IP_ADDRESS}

Immediate investigation:
1. Identify what systems/accounts were accessed
2. Check all logs for this IP in last 24 hours
3. Verify if IP belongs to legitimate user/service
4. Check for data exfiltration signs
5. Recommend immediate containment actions
6. Preserve evidence (logs, connection data)

Report findings immediately. Do NOT make changes without approval.
```

### Complete System Failure
```
CRITICAL: {SYSTEM} is completely unresponsive.

Emergency diagnostic:
1. Can the system be pinged?
2. Can we SSH/access console?
3. If accessible: Check system load, disk, memory
4. If not accessible: Likely need physical intervention
5. Check for recent changes or events
6. Assess impact on other systems
7. Provide recovery options (restore, rebuild, wait)

Report findings and recommend next steps immediately.
```

## Integration Examples

### Example 1: Discord Bot Command Handler
```python
# In Discord bot (bot-buttons.py or similar)

@bot.command(name='infra')
async def infrastructure_check(ctx, *args):
    """Check infrastructure health"""

    # Parse command: !infra check sccm
    # Or: !infra bandwidth
    # Or: !infra status all

    if args[0] == 'check' and args[1] == 'sccm':
        prompt = "Check SCCM server health: services, disk, errors"
    elif args[0] == 'bandwidth':
        prompt = "Analyze current network bandwidth usage and top consumers"
    elif args[0] == 'status' and args[1] == 'all':
        prompt = "Quick status check: Unraid, SCCM, DC, Network - brief format"

    # Execute via SSH to Unraid (where Claude Code runs)
    result = subprocess.run(
        ['ssh', 'root@<ADD-IP-ADDRESS>', f'claude ask "{prompt}"'],
        capture_output=True, text=True
    )

    # Send response to Discord
    await ctx.send(f"```{result.stdout}```")
```

### Example 2: n8n Node Configuration
```json
{
  "nodes": [
    {
      "name": "Schedule - Hourly Check",
      "type": "n8n-nodes-base.cron",
      "position": [100, 200],
      "parameters": {
        "triggerTimes": {
          "item": [
            {"mode": "everyHour"}
          ]
        }
      }
    },
    {
      "name": "Claude Infrastructure Check",
      "type": "n8n-nodes-base.ssh",
      "position": [300, 200],
      "parameters": {
        "command": "claude ask 'Quick infrastructure health check - critical services only'",
        "host": "<ADD-IP-ADDRESS>",
        "authentication": "privateKey",
        "privateKey": "={{$credentials.ssh_claude_key}}"
      }
    },
    {
      "name": "Check for Issues",
      "type": "n8n-nodes-base.if",
      "position": [500, 200],
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json.stdout}}",
              "operation": "contains",
              "value2": "⚠️|❌"
            }
          ]
        }
      }
    },
    {
      "name": "Send Alert",
      "type": "n8n-nodes-base.discord",
      "position": [700, 200],
      "parameters": {
        "webhook": "={{$credentials.discord_infra_webhook}}",
        "message": "Infrastructure Alert:\n```{{$json.stdout}}```"
      }
    }
  ]
}
```

## Tips for Effective Automation

1. **Start Simple**: Begin with read-only monitoring before automating fixes
2. **Test Thoroughly**: Validate prompts manually before scheduling
3. **Set Boundaries**: Clearly define what Claude can auto-fix vs. alert
4. **Log Everything**: Keep audit trail of all automated actions
5. **Fail Safely**: Always have fallback to manual intervention
6. **Monitor the Monitor**: Alert if automation workflows fail
7. **Iterate**: Refine prompts based on results
8. **Document**: Keep this guide updated with new working prompts
9. **Context Matters**: Reference specific context docs in prompts
10. **Human in Loop**: For critical changes, always require approval

## Troubleshooting Automation Issues

### Claude Not Responding
```
Check:
1. Is Unraid server accessible?
2. Is claude-code container running?
3. Check container logs: docker logs claude-code
4. Verify SSH key permissions
5. Test manual command: ssh root@<ADD-IP-ADDRESS> claude ask "test"
```

### Incorrect/Incomplete Responses
```
Improve prompts by:
1. Being more specific about what data to collect
2. Explicitly requesting structured output format
3. Referencing specific context documents
4. Setting clear success criteria
5. Requesting step-by-step actions
```

### Automation Acting Unexpectedly
```
Safety measures:
1. Review automation logs
2. Check if prompt was misinterpreted
3. Add explicit constraints to prompts
4. Use "recommend only, don't execute" mode
5. Implement approval gates for critical actions
```

## Next Steps

1. **Test Prompts**: Try examples via Discord bot to validate
2. **Create Workflows**: Build n8n workflows for highest priority monitors
3. **Refine Context**: Update infrastructure docs as systems change
4. **Expand Coverage**: Add prompts for File Server, Exchange when configured
5. **Build History**: Track what prompts work best, document lessons learned
6. **Share**: Contribute successful prompts back to this guide

## Feedback and Improvements

As you use these templates, note:
- Which prompts work best
- Which need refinement
- New scenarios to automate
- Edge cases encountered
- Performance optimizations

Update this guide regularly to reflect real-world learnings.
