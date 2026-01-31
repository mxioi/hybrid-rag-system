# Infrastructure Context Documentation

## Overview

This directory contains comprehensive context documentation for all infrastructure components in the homelab. These documents enable Claude Code to intelligently monitor, troubleshoot, and manage infrastructure through Discord, n8n workflows, or direct interaction.

## Documentation Structure

### System-Specific Context
Each system has a dedicated context document with:
- System overview and specifications
- Common tasks and commands
- Monitoring points and health checks
- Troubleshooting scenarios
- Automation prompt templates
- Integration examples

### Available Documents

#### 1. **unraid-server.md** - Primary NAS/Hypervisor Host
**System**: <ADD-HOSTNAME> (<ADD-IP-ADDRESS>)
**Purpose**: Docker container host, file shares, VM hypervisor
**Key Services**: lldap, netdata, Grafana, Plex, CrowdSec, Lancache, PhotoPrism
**Coverage**:
- Array and disk management
- Docker container operations
- Storage monitoring
- Network configuration
- Log analysis
- Backup management

**Quick Access**:
```bash
cat /path/to/hybrid-rag-system/context/infrastructure/unraid-server.md
```

**Example Prompt**:
```
Check my Unraid server status:
1. Array status and disk health
2. Critical containers running
3. Disk space on array and cache
4. Recent CrowdSec alerts
```

---

#### 2. **sccm-server.md** - Configuration Management
**System**: VM3-APP1 (10.0.0.X)
**Purpose**: Microsoft Endpoint Configuration Manager (MECM/SCCM)
**Key Services**: SMS_*, WSUS, SQL Server, IIS
**Coverage**:
- SCCM service health
- Client management
- Content distribution
- Patch management (WSUS)
- Database maintenance
- Log file monitoring

**Monitoring Script**:
```bash
/path/to/hybrid-rag-system/context/tools/check-sccm.sh
```

**Example Prompt**:
```
Check SCCM server health:
1. All SMS_* services running?
2. Disk space on C: drive
3. Any critical errors in last 24h
4. WSUS last sync status
5. Active client count
```

---

#### 3. **domain-controller.md** - Active Directory
**System**: VM1-DC1 (<ADD-IP-ADDRESS>) + VM5-DC2 (10.0.0.X)
**Purpose**: Domain services, DNS, authentication, Group Policy
**Key Services**: NTDS, DNS, ADWS, KDC, Netlogon
**Coverage**:
- AD health and replication
- User and computer management
- DNS management
- Group Policy operations
- FSMO role management
- Security auditing

**Known Issue**: DC2 replication failure since Nov 18, 2025

**Example Prompt**:
```
Check Active Directory health:
1. NTDS and DNS services running?
2. AD replication status
3. Any authentication failures (last 24h)
4. Locked out accounts
5. Time synchronization
```

---

#### 4. **udm-pro-max.md** - Network Infrastructure
**System**: <ADD-HOSTNAME> (<ADD-IP-ADDRESS>)
**Purpose**: Network gateway, firewall, UniFi controller, router
**Key Services**: WAN routing, WiFi controller, IPS/IDS, VPN
**Coverage**:
- Network health monitoring
- Access point management
- Client connectivity
- Bandwidth analysis
- Firewall and security
- WAN failover

**API Integration**: Full UniFi API access with key
**Monitoring Script**:
```bash
/path/to/hybrid-rag-system/context/tools/check-udm.sh
```

**Example Prompt**:
```
Check network status:
1. WAN connectivity
2. Access point health and client counts
3. Total connected clients
4. Any active network alerts
5. Top bandwidth consumers
```

---

#### 5. **automation-guide.md** - Comprehensive Automation Reference
**Purpose**: Prompt templates, workflow examples, best practices
**Coverage**:
- Discord bot interaction prompts
- n8n workflow templates
- Advanced automation scenarios
- Prompt engineering best practices
- Emergency response procedures
- Integration examples

**Sections Include**:
- 50+ ready-to-use prompt templates
- 7 complete n8n workflow examples
- 5 advanced automation scenarios
- Emergency response playbooks
- Integration code examples
- Troubleshooting guides

**Quick Access**:
```bash
cat /path/to/hybrid-rag-system/context/infrastructure/automation-guide.md
```

## How to Use This Documentation

### For Manual Tasks
1. Open relevant system context document
2. Navigate to "Common Tasks" section
3. Find the command or procedure needed
4. Execute via SSH or appropriate interface

### For Discord Interactions
1. Reference automation-guide.md for prompt templates
2. Customize prompt for your specific need
3. Send to Discord bot: `!ask [your prompt]`
4. Claude will reference appropriate context automatically

### For n8n Automation
1. Review workflow templates in automation-guide.md
2. Adapt to your specific monitoring needs
3. Import/create workflow in n8n
4. Configure schedule and notification channels
5. Test thoroughly before production use

### For Emergency Response
1. Identify affected system
2. Open relevant context document
3. Navigate to "Troubleshooting Scenarios"
4. Follow diagnostic steps
5. Use emergency prompts from automation-guide.md if needed

## Integration Points

### SSH Access
All Windows servers accessible via:
```bash
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> \
  svc-monitor@[IP_ADDRESS]
```

### Monitoring Scripts
Located in: `/path/to/hybrid-rag-system/context/tools/`
- `check-sccm.sh` - SCCM health check
- `check-proxmox.sh` - Proxmox resources
- `check-udm.sh` - Network health

### Configuration Files
Located in: `/path/to/hybrid-rag-system/context/credentials/`
- `sccm-config.env` - SCCM connection details
- `proxmox-config.env` - Proxmox API credentials
- `udm-config.env` - UniFi API key and endpoints
- `<ADD-SSH-KEY>` - SSH private key (permissions 600)

## Quick Reference Commands

### Check All Systems
```bash
# Via monitoring scripts
/path/to/hybrid-rag-system/context/tools/check-sccm.sh
/path/to/hybrid-rag-system/context/tools/check-proxmox.sh
/path/to/hybrid-rag-system/context/tools/check-udm.sh
```

### Manual SSH Access
```bash
# SCCM
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> svc-monitor@10.0.0.X

# Domain Controller
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> svc-monitor@<ADD-IP-ADDRESS>

# Proxmox
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> claude-monitor@10.0.0.X

# UDM
ssh -i /path/to/hybrid-rag-system/context/credentials/<ADD-SSH-KEY> root@<ADD-IP-ADDRESS>
```

### API Access Examples
```bash
# UniFi Network Health
curl -k -X GET 'https://<ADD-IP-ADDRESS>/proxy/network/api/s/default/stat/health' \
  -H "X-API-KEY: SDSIPtAPeeA_lAV9C6u_LVLbZ57wnMEw"

# Proxmox (via SSH - token for web UI only)
ssh claude-monitor@10.0.0.X "pvesh get /nodes/proxmox/status"
```

## Maintenance

### Updating Documentation
When infrastructure changes:
1. Update relevant system context document
2. Update automation-guide.md if new scenarios
3. Update this README if structure changes
4. Test updated prompts/commands
5. Document changes in commit message

### Adding New Systems
For new infrastructure components:
1. Create new context document following existing format
2. Include: overview, tasks, monitoring, troubleshooting, prompts
3. Add monitoring script to tools/ directory
4. Add credentials to credentials/ directory (secure permissions!)
5. Update this README with new system details
6. Add integration examples to automation-guide.md

### Regular Reviews
**Monthly**:
- Verify all commands still work
- Update system specifications if changed
- Review and refine prompt templates based on usage
- Check for outdated information

**Quarterly**:
- Comprehensive test of all monitoring scripts
- Review and update troubleshooting scenarios
- Audit access credentials and rotate if needed
- Update workflow templates with improvements

## Security Considerations

All documentation follows security best practices:
- **Read-only Access**: Service accounts have minimal permissions
- **SSH Keys Only**: No password authentication
- **Secure Storage**: Credentials in protected directory (600 permissions)
- **Audit Trail**: All access logged
- **Principle of Least Privilege**: Each account limited to required scope
- **No Secrets in Docs**: API keys referenced, not embedded (except in config files)

### Service Accounts
- **svc-monitor**: Windows servers (SCCM, DC)
  - Member of: Event Log Readers, Performance Monitor Users
  - Cannot modify system configuration
- **claude-monitor@pve**: Proxmox
  - Role: PVEAuditor (read-only)
- **root@udm**: UDM Pro Max
  - Full access (owner decision)

## Support and Troubleshooting

### If Documentation is Unclear
1. Check automation-guide.md for examples
2. Review "Common Tasks" section in system doc
3. Test commands manually before automating
4. Document improvements and update docs

### If Commands Don't Work
1. Verify SSH access to target system
2. Check service account permissions
3. Review recent system changes
4. Check system-specific troubleshooting section
5. Verify context document is up-to-date

### If Automation Fails
1. Review automation-guide.md troubleshooting section
2. Test prompt manually via Discord
3. Check n8n workflow logs
4. Verify claude-code container is running
5. Check SSH key permissions (600)

## Additional Resources

### Microsoft Documentation
- SCCM: https://docs.microsoft.com/mem/configmgr/
- Active Directory: https://docs.microsoft.com/windows-server/identity/ad-ds/

### UniFi Documentation
- UniFi API: https://unifi.ui.com/consoles/[ID]/unifi-api/network
- Community: https://community.ui.com/

### Unraid Resources
- Forums: https://forums.unraid.net/
- Documentation: https://docs.unraid.net/

### Proxmox Documentation
- Official: https://pve.proxmox.com/pve-docs/
- API: https://pve.proxmox.com/pve-docs/api-viewer/

## Future Enhancements

### Planned Documentation
- File Server context (when configured)
- Exchange Server context (when configured)
- Backup/DR procedures document
- Network topology diagram
- Dependency mapping

### Planned Automation
- Automated daily health reports
- Proactive disk space management
- Intelligent service recovery
- Cross-system correlation analysis
- Predictive failure detection

### Planned Integrations
- Grafana dashboard integration
- Centralized logging (ELK/Graylog)
- Alert escalation workflows
- Mobile app notifications
- Voice assistant integration

---

**Last Updated**: 2025-12-17
**Maintainer**: Claude Code + User
**Version**: 1.0

For questions or improvements, update this documentation and commit changes to preserve institutional knowledge.
