import numpy as np
import os
from multiprocessing import Process
import subprocess
import sys
import math

PATH = "train_data"

cores = [0,1,2,3,4,5,6,7]
benchmarks = [6, 8, 5,5,5,5,5,5]

# names=(
#         "b600" "b602" "b603" "b605" "b607" "b619" "b620" "b621" "b623" "b625" "b627" "b628" "b631" "b638" "b641" "b644" "b648" "b649" "b654" "b657"
#         "b500" "b502" "b503" "b505" "b507" "b508" "b510" "b511" "b519" "b520" "b521" "b523" "b525" "b526" "b527" "b531" "b538" "b541" "b544" "b548" "b549" "b554" "b557"
# )
spec_names = ["perlbench","sgcc","bwaves","mcf","cactuBSSN","lbm","omnetpp","wrf","xalancbmk","x264","cam4","pop2","deepsjeng","imagick","leela","nab","exchange","fotonik","sroms","xz",
            "perlbench","cpugcc","bwaves","mcf","cactusBSSN","namd","parest","povray","lbm","omnetpp","wrf","cpuxalan","x264","blender","cam4","deepsjeng","imagick","leela","nab","exchange","fotonik","roms","xz"]


#200ms perf
minvalues = {'r0C0':43000000, 'r03C':400000000, 'r4F2E':520000, 'r412E':35000, 'r02A3':5000000, 'r01A3':9400000}
maxvalues = {'r0C0':3100000000, 'r03C':1500000000, 'r4F2E':92000000, 'r412E':34000000, 'r02A3':1750000000, 'r01A3':850000000}
ranges = {}
for key in minvalues.keys():
    ranges[key]=maxvalues[key]-minvalues[key]


def launcher(core_idx, benchmark):
    cmd = "sudo taskset -c " + str(core_idx) + " ./perf/shell.sh " + str(benchmark)
    os.system(cmd)

class CATEnv:
    def __init__(self, perf_period):
        self.perf_period = perf_period

        cmd = "sudo pqos -R"
        os.system(cmd)
        cmd = "sudo pqos -a \"llc:1=0,1;llc:2=2-7\""    # suppose clustering is done
        os.system(cmd)

        self.way = [11,11]
        self.clos = [0,0,1,1,1,1,1,1]
        self.core = [[0,1],[2,3,4,5,6,7]]   # which cores does each clos(idx) has

        self.state = np.zeros(6*8)
        for ii, i in enumerate([0,6,12,18,24,30,36,42]):
            self.state[i] = self.way[self.clos[ii]] / 11

        # run benchmarks
        for ii,core_idx in enumerate(cores):
            # Thread(target=launcher, args=(core_idx, benchmarks[ii],)).start()
            Process(target=launcher, args=(core_idx, benchmarks[ii],)).start()


        # self.ipcdiff = []
        self.cycles_l2_miss = []
        self.ipc = []
        for _ in range(len(cores)):
            self.cycles_l2_miss.append([])
            self.ipc.append([])

    def reset(self):
        cmd = "sudo pqos -R"
        os.system(cmd)
        cmd = "sudo pqos -a \"llc:1=0,1;llc:2=2-7\""
        os.system(cmd)

        self.way = [11,11]
        self.clos = [0,0,1,1,1,1,1,1]
        self.core = [[0,1],[2,3,4,5,6,7]]

        self.state = np.zeros(6*8)
        for ii, i in enumerate([0,6,12,18,24,30,36,42]):
            self.state[i] = self.way[self.clos[ii]] / 11

        return self.state

    
    def step(self, action):
        if action <=3:
            if action <=1:  # 0,1 (clos1 -,+)
                clos_idx=1
                clos_op = 0 if action==0 else 1
            elif action <=3:   # 2,3 (clos2 -,+)
                clos_idx=2
                clos_op = 0 if action==2 else 1

            flag = True
            if clos_op == 0:    # way -1
                if self.way[clos_idx-1] > 1:
                    self.way[clos_idx-1] -= 1
                    for c in self.core[clos_idx-1]:
                        self.state[c*6] = self.way[clos_idx-1] / 11
                else: flag = False    # there is no way left to minus 1
            else:   # clos_op==1   # way + 1
                if self.way[clos_idx-1] < 11:
                    self.way[clos_idx-1] += 1
                    for c in self.core[clos_idx-1]:
                        self.state[c*6] = self.way[clos_idx-1] / 11
                else:
                    flag = False    # there is no way left to plus 1
            
            if flag==True:
                if clos_idx == 1:   # CLOS1 : filling 1 from left
                    clos_ary = [1]*(self.way[clos_idx-1]) + [0]*(11-self.way[clos_idx-1])
                else :              # CLOS2 : filling 1 from right
                    clos_ary = [0]*(11-self.way[clos_idx-1]) + [1]*(self.way[clos_idx-1])
                clos_val = ''.join(str(e) for e in clos_ary)
                clos_val = hex(int(clos_val, 2))
                cmd = "sudo pqos -e \"llc:"+str(clos_idx)+"="+str(clos_val)+"\""
                print(cmd)
                os.system(cmd)
        
        elif action == 4 : print("nothing")


        # PERF_CMD = "sudo perf stat -a --per-core -x, --time 100 -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 2>&1"
        PERF_CMD = "sudo perf stat -a --per-core -x, --time "+self.perf_period+" -e r0C0,r03C,r4F2E,r412E,r01A3,r02A3 2>&1"
        t = subprocess.check_output(PERF_CMD, shell=True)
        t = t.decode()

        pmu_dicts = [] # for unnormalized value
        state_idx = 0
        for line in t.split('\n'):
            if line=='':break
            # modify depending on server
            elems = line.split(",")
            temp = elems[0].split("C")
            temp = int(temp[-1])
            cpuCore = temp+int(elems[0][1])*16
            val = int(elems[2])
            evt = str(elems[4])

            if cpuCore <= 7:
                if evt == "r0C0":
                    state_idx=1
                    pmu_dicts.append(dict())
                pmu_dicts[cpuCore][evt] = val
                
                if evt == "r01A3":continue  # cycles_l2_miss does not need to be normalized.. Just for another reward
                # normalize val
                if val < minvalues[evt]:
                    print('{} {} out of range {}~{}'.format(evt, val, minvalues[evt], maxvalues[evt]))
                    val = minvalues[evt]
                if val > maxvalues[evt]:
                    print('{} {} out of range {}~{}'.format(evt, val, minvalues[evt], maxvalues[evt]))
                    val = maxvalues[evt]
                val = (val-minvalues[evt])/ranges[evt]

                self.state[cpuCore*6+state_idx] = val
                state_idx+=1

        
        # Calculate Reward
        sum_reward = 0
        for core_idx in range(len(cores)):
            ipc = pmu_dicts[core_idx]["r0C0"]/pmu_dicts[core_idx]["r03C"]
            cl2m = pmu_dicts[core_idx]["r01A3"]

            sum_reward += ipc

            self.ipc[core_idx].append(ipc)     # save data
            self.cycles_l2_miss[core_idx].append(cl2m)  # save data

        return self.state, sum_reward


    def finish(self):
        print("End of procedure")
        # DEF_KILL_CMD = "bash -c 'sudo killall -9 shell.sh'"
        DEF_KILL_CMD = "sudo killall -9 shell.sh"
        os.system(DEF_KILL_CMD)
        DEF_KILL_CMD = "sudo kill -9 $(pgrep '{}|{}|{}')".format(spec_names[5],spec_names[6],spec_names[8])
        os.system(DEF_KILL_CMD)

        os.makedirs(PATH+"/perf", exist_ok=True)
        for i in range(8):
            np.savetxt(PATH+"/perf/ipc_"+str(i)+".txt", self.ipc[i], delimiter=',')
            np.savetxt(PATH+"/perf/cycles_l2_miss_"+str(i)+".txt", self.cycles_l2_miss[i], delimiter=',')