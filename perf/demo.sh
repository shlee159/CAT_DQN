#!/bin/bash

#cycles_l2_miss : r01A3
sudo perf stat -a --per-core -I $1 -x, -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 python3 ./perf/launcher.py 2>&1