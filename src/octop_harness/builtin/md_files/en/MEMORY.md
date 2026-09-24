---
summary: "Agent long-term memory — tool config and accumulated wisdom"
read_when:
  - Bootstrapping a workspace manually
---

## Tool Config

Skills tell you how tools work. This file records **your environment** — the specifics that belong to this user and this machine only.

### What to Write Here

Anything that helps you hit the ground running next time you wake up. Treat this as your quick-reference manual.

Examples:

- SSH connection details and aliases
- User-side config used when executing Skills
- Device names, paths, ports, and other local specifics

### Format Reference

```markdown
### SSH

- home-server → 192.168.1.100, user: admin, port: 22
- dev-box → 10.0.0.5, user: deploy, key: ~/.ssh/dev_rsa
```
