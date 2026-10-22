#!/bin/bash
set -euo pipefail
# Dedicated port; all keys additionally isolated by experiment ID.
if ! redis-cli -h 10.10.4.1 -p 16389 ping >/dev/null 2>&1; then
 redis-server --bind 127.0.0.1 10.10.4.1 --port 16389 --protected-mode no --save '' --appendonly no --daemonize yes --pidfile /workspace/flash2/results/redis.pid --logfile /workspace/flash2/results/redis.log
fi
