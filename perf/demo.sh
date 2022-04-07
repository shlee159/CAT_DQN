#!/bin/bash
# sudo perf stat -a --per-core -I 200 -x, -e r0C0,r03C,r4F2E,r412E,r02A3 python3 ./perf/launcher.py 2>&1


# add cycles_l2_miss(r01A3)
# sudo perf stat -a --per-core -I 200 -x, -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 python3 ./perf/launcher.py 2>&1
# sudo perf stat -a --per-core -I 100 -x, -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 python3 ./perf/launcher.py 2>&1
sudo perf stat -a --per-core -I 400 -x, -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 python3 ./perf/launcher.py 2>&1