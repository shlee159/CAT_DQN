from threading import Thread
import os

def launcher(core_idx, benchmark):
    cmd = "sudo taskset -c " + str(core_idx) + " ./perf/shell.sh " + str(benchmark)
    os.system(cmd)


for ii,core_idx in enumerate([0,1,2,3,4,5,6,7]):
    benchmarks = [6, 8, 5,5,5,5,5,5]
    Thread(target=launcher, args=(core_idx, benchmarks[ii],)).start()
