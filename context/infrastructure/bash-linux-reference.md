# Bash & Linux Reference for Local LLM

Quick reference for common commands across Unraid, Proxmox, and Linux systems.

## Unraid Commands (<ADD-IP-ADDRESS>)

### Docker Management
```bash
# List containers
docker ps -a
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# Container lifecycle
docker start <container>
docker stop <container>
docker restart <container>

# Logs
docker logs -f <container>          # Follow logs
docker logs --tail 100 <container>  # Last 100 lines

# Stats
docker stats --no-stream

# Execute in container
docker exec -it <container> /bin/bash
docker exec -it <container> sh

# Inspect
docker inspect <container> | jq '.[0].NetworkSettings.IPAddress'

# Cleanup
docker system prune -a              # Remove unused images/containers
docker volume prune                 # Remove unused volumes
```

### Disk & Storage
```bash
# Disk usage
df -h
df -h /mnt/disk1 /mnt/disk2 /mnt/cache

# Directory sizes
du -sh /mnt/user/appdata/*
du -sh /mnt/user/* | sort -h

# Array status
cat /proc/mdstat
mdcmd status

# Find large files
find /mnt/user -type f -size +1G -exec ls -lh {} \;

# Check SMART
smartctl -a /dev/sda
smartctl -H /dev/sda              # Health only
```

### Process Management
```bash
# Find process
ps aux | grep <name>
pgrep -a <name>

# Kill process
kill <pid>
kill -9 <pid>                     # Force kill
pkill <name>

# Resource usage
top -bn1 | head -20
htop                              # Interactive (if installed)
free -h                           # Memory
uptime                            # Load average
```

### Network
```bash
# IP info
ip addr show
ip -brief addr

# Connections
ss -tulpn                         # Listening ports
netstat -tulpn                    # Alternative

# Test connectivity
ping -c 3 10.0.0.X
curl -s http://localhost:8000/api/v2/heartbeat

# DNS
nslookup homelab.local
dig homelab.local
```

### Logs
```bash
# System logs
tail -f /var/log/syslog
dmesg | tail -50

# Docker container logs location
ls /mnt/user/appdata/*/logs/
```

---

## Proxmox Commands (<ADD-IP-ADDRESS>)

### VM Management
```bash
# List VMs
qm list

# VM lifecycle
qm start <vmid>
qm stop <vmid>
qm shutdown <vmid>                # Graceful
qm reboot <vmid>

# VM info
qm status <vmid>
qm config <vmid>

# Console access
qm terminal <vmid>
```

### LXC Container Management
```bash
# List containers
pct list

# Container lifecycle
pct start <ctid>
pct stop <ctid>
pct shutdown <ctid>
pct reboot <ctid>

# Enter container
pct enter <ctid>

# Execute command in container
pct exec <ctid> -- <command>
pct exec 300 -- ollama list

# Container info
pct status <ctid>
pct config <ctid>
```

### Storage
```bash
# List storage
pvesm status

# ZFS commands
zpool status
zpool list
zfs list

# Backup info
ls /mnt/pve/pbs-storage/
```

### Cluster/Node
```bash
# Node status
pvesh get /nodes
pveversion

# Resources
pvesh get /nodes/proxmox/status
```

---

### SSH with specific key
```bash
ssh -i /path/to/key user@host "command"
```

---

## File Operations

### Find & Search
```bash
# Find files
find /path -name "*.yaml"
find /path -name "*.log" -mtime -1    # Modified last 24h
find /path -type f -size +100M        # Files > 100MB

# Search in files (ripgrep preferred)
rg "pattern" /path
rg -i "docker" /mnt/user/appdata/     # Case insensitive
grep -r "pattern" /path               # Fallback
```

### File Management
```bash
# Copy with progress
rsync -avh --progress source/ dest/

# Secure copy
scp file.txt user@host:/path/
scp -r folder/ user@host:/path/

# Archive
tar -czvf archive.tar.gz /path/
tar -xzvf archive.tar.gz

# Permissions
chmod 755 script.sh
chmod -R 644 /path/files/
chown user:group file
```

---

## System Administration

### Service Management
```bash
# Systemd
systemctl status <service>
systemctl start/stop/restart <service>
systemctl enable/disable <service>
journalctl -u <service> -f

# Check all failed
systemctl --failed
```

### Cron Jobs
```bash
# Edit crontab
crontab -e

# List cron jobs
crontab -l

# Common patterns
# Min Hour Day Month Weekday Command
# 0    2    *   *     *       /path/script.sh  # Daily at 2am
# */5  *    *   *     *       /path/script.sh  # Every 5 min
```

### Users & Permissions
```bash
# Current user
whoami
id

# Switch user
su - username
sudo -u username command
```

---

## Troubleshooting

### Container won't start
```bash
docker logs <container>
docker inspect <container> | jq '.[0].State'
docker events --since 10m
```

### Disk full
```bash
df -h
du -sh /* 2>/dev/null | sort -h
docker system df                  # Docker disk usage
```

### Network issues
```bash
ping gateway
ip route
cat /etc/resolv.conf
curl -v http://service:port
```

### Process using port
```bash
ss -tulpn | grep :8080
lsof -i :8080
```
