import numpy as np
import os
import threading
# from multiprocessing import Process
import subprocess
import sys
import math


# names=(
#         "b600" "b602" "b603" "b605" "b607" "b619" "b620" "b621" "b623" "b625" "b627" "b628" "b631" "b638" "b641" "b644" "b648" "b649" "b654" "b657"
#         "b500" "b502" "b503" "b505" "b507" "b508" "b510" "b511" "b519" "b520" "b521" "b523" "b525" "b526" "b527" "b531" "b538" "b541" "b544" "b548" "b549" "b554" "b557"
# )
spec_names = ["perlbench","sgcc","bwaves","mcf","cactuBSSN","lbm","omnetpp","wrf","xalancbmk","x264","cam4","pop2","deepsjeng","imagick","leela","nab","exchange","fotonik","sroms","xz",
            "perlbench","cpugcc","bwaves","mcf","cactusBSSN","namd","parest","povray","lbm","omnetpp","wrf","cpuxalan","x264","blender","cam4","deepsjeng","imagick","leela","nab","exchange","fotonik","roms","xz"]


#100ms perf
minvalues = {'r0C0':22000000, 'r03C':150000000, 'r4F2E':300000, 'r412E':27000, 'r02A3':2800000, 'r01A3':4800000}
maxvalues = {'r0C0':1900000000, 'r03C':800000000, 'r4F2E':58000000, 'r412E':24000000, 'r02A3':1100000000, 'r01A3':620000000}

#200ms perf
minvalues = {'r0C0':43000000, 'r03C':400000000, 'r4F2E':520000, 'r412E':35000, 'r02A3':5000000, 'r01A3':9400000}
maxvalues = {'r0C0':3100000000, 'r03C':1500000000, 'r4F2E':92000000, 'r412E':34000000, 'r02A3':1750000000, 'r01A3':850000000}
ranges = {}
for key in minvalues.keys():
    ranges[key]=maxvalues[key]-minvalues[key]


def perf_runner(cmd, self):
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)

    for line in iter(proc.stdout.readline, ''):
        line = line.decode()
        if line=='':break
        if "Killed" in line:
            continue
        line = line.strip()
        elems = line.split(",")

        time = float(elems[0])
        # print(time)
        temp = elems[1].split("C")
        temp = int(temp[-1])
        cpuCore = temp+int(elems[1][1])*16
        val = int(elems[3])
        evt = str(elems[5])

        self.digest(time, cpuCore, val ,evt)


class CATEnv():
    def __init__(self, perf_period):
        self.perf_period = perf_period

        cmd = "sudo pqos -R"
        os.system(cmd)
        # Intel CAT CLOS default
        self.way = [11,11]
        self.clos = [0,0,1,1,1,1,1,1]
        self.core = [[0,1],[2,3,4,5,6,7]]

        self.state = np.zeros(6*8)

        cmd = "sudo pqos -a \"llc:1=0,1;llc:2=2-7\""
        os.system(cmd)
        ## 
        self.way = [6,6]
        cmd = "sudo pqos -e \"llc:1=0x7e0;llc:2=0x03f\""
        os.system(cmd)

        for ii, i in enumerate([0,6,12,18,24,30,36,42]):
            self.state[i] = self.way[self.clos[ii]] / 11

        ##
        self.step_flag=False
        self.pmu_dicts=[]
        self.prev_time=None
        self.state_idx=0

        self.prev_reward=0

        self.readingline = False
        self.step_flag = False


    def reset(self):
        cmd = "sudo pqos -R"
        os.system(cmd)
        # Intel CAT CLOS default
        self.way = [11,11]
        self.clos = [0,0,1,1,1,1,1,1]
        self.core = [[0,1],[2,3,4,5,6,7]]

        self.state = np.zeros(6*8)

        cmd = "sudo pqos -a \"llc:1=0,1;llc:2=2-7\""
        os.system(cmd)
        ## 
        self.way = [6,6]
        cmd = "sudo pqos -e \"llc:1=0x7e0;llc:2=0x03f\""
        os.system(cmd)

        for ii, i in enumerate([0,6,12,18,24,30,36,42]):
            self.state[i] = self.way[self.clos[ii]] / 11

        return self.state
    
    def run(self, callback):
        self.callback = callback

        # DEF_EXEC_CMD = "/home/cell/sh/CAT/perf/demo.sh"
        DEF_EXEC_CMD = "sudo taskset -c 31 ./perf/demo.sh " + str(self.perf_period)
        self.t = threading.Thread(target=perf_runner, args=(DEF_EXEC_CMD, self))
        self.t.start()
        # Process(target=perf_runner, args=(DEF_EXEC_CMD, self)).start()

    def step(self, action):
        if action <=3:
            if action <=1:  # 0,1 (clos1 -,+)
                clos_idx=1
                clos_op = 0 if action==0 else 1
            elif action <=3:   # 2,3 (clos2 -,+)
                clos_idx=2
                clos_op = 0 if action==2 else 1

            flag=True
            if clos_op==0: # way - 1
                if self.way[clos_idx-1] > 1:
                    self.way[clos_idx-1] -= 1
                    #
                    for c in self.core[clos_idx-1]:
                        self.state[c*6] = self.way[clos_idx-1] / 11
                else: flag=False    # there is no way left to minus 1
            else:   # clos_op==1   # way + 1
                if self.way[clos_idx-1] < 11:
                    self.way[clos_idx-1] += 1
                    #
                    for c in self.core[clos_idx-1]:
                        self.state[c*6] = self.way[clos_idx-1] / 11
                else:
                    flag=False  # there is no way left to plus 1
            if flag==True:
                if clos_idx==1: # CLOS 1 update filling 0 from the end
                    clos_ary = [1]*(self.way[clos_idx-1]) + [0]*(11-self.way[clos_idx-1])
                else :          # CLOS 2 update filling 0 from the start
                    clos_ary = [0]*(11-self.way[clos_idx-1]) + [1]*(self.way[clos_idx-1])
                clos_val = ''.join(str(e) for e in clos_ary)
                clos_val = hex(int(clos_val, 2))
                cmd = "sudo pqos -e \"llc:"+str(clos_idx)+"="+str(clos_val)+"\""
                print(cmd)
                os.system(cmd)

        # else: print("stay action")  # action 4
        ##
        elif action == 4: print("stay action") 

        # else:
        #     if action <= 6:
        #         clos_idx = 1
        #         clos_op = 0 if action==5 else 1
        #     elif action <= 8:
        #         clos_idx = 2
        #         clos_op = 0 if action==7 else 1
            
        #     flag=True
        #     if clos_op==0:
        #         if self.way[clos_idx-1] > 2:
        #             self.way[clos_idx-1] -= 2
        #             #
        #             for c in self.core[clos_idx-1]:
        #                 self.state[c*6] = self.way[clos_idx-1] / 11
        #         else: flag=False
        #     else:
        #         if self.way[clos_idx-1] < 10:
        #             self.way[clos_idx-1] += 2
        #             #
        #             for c in self.core[clos_idx-1]:
        #                 self.state[c*6] = self.way[clos_idx-1] / 11
        #         else: flag=False
            
        #     if flag==True:
        #         if clos_idx==1: # CLOS 1 update filling 0 from the end
        #             clos_ary = [1]*(self.way[clos_idx-1]) + [0]*(11-self.way[clos_idx-1])
        #         else :          # CLOS 2 update filling 0 from the start
        #             clos_ary = [0]*(11-self.way[clos_idx-1]) + [1]*(self.way[clos_idx-1])
        #         clos_val = ''.join(str(e) for e in clos_ary)
        #         clos_val = hex(int(clos_val, 2))
        #         cmd = "sudo pqos -e \"llc:"+str(clos_idx)+"="+str(clos_val)+"\""
        #         print(cmd)
        #         os.system(cmd)
        



        # when step function starting, if perf is being read, wait until it finishes.
        while self.readingline==True: continue
        # Now, readingline == False
        self.step_flag=True
        while self.step_flag==True: continue

        sum_reward = 0
        for core_idx in range(8):
            reward = self.pmu_dicts[core_idx]["r0C0"]/self.pmu_dicts[core_idx]["r03C"]
            # reward = math.pow(reward,2)
            # reward = math.pow(reward,3)
            # reward = math.pow(reward,4)
            sum_reward += reward
        # sum_reward /= 8
        # sum_reward = math.pow(sum_reward, 2)

        return self.state, sum_reward

    def digest(self, time, cpuCore, val, evt):
        if self.prev_time != time:
            if self.prev_time is not None:
                self.callback(self.pmu_dicts)
            self.pmu_dicts = []

            self.readingline = True

        if len(self.pmu_dicts)<=cpuCore:
            self.pmu_dicts.append(dict())
            self.pmu_dicts[cpuCore]["time"] = time
            self.state_idx=1
        self.pmu_dicts[cpuCore][evt] = val

        if evt == "r01A3":return

        if self.step_flag == True and cpuCore<=7:
            #/->
            if val < minvalues[evt]: 
                print('{} {} out of range {}~{}'.format(evt, val, minvalues[evt], maxvalues[evt]))
                val = minvalues[evt]
            if val > maxvalues[evt]: 
                print('{} {} out of range {}~{}'.format(evt, val, minvalues[evt], maxvalues[evt]))
                val = maxvalues[evt]
            val = (val-minvalues[evt])/ranges[evt]  # normalization
            #<-/
            self.state[cpuCore*6+self.state_idx] = val
        if cpuCore==31 and evt=='r02A3':
            self.readingline = False
            self.step_flag = False
        self.state_idx += 1

        self.prev_time = time 

    def finish(self):
        print("End of procedure")

        DEF_KILL_CMD = "bash -c 'sudo killall -9 shell.sh'"
        os.system(DEF_KILL_CMD)

        DEF_KILL_CMD = "sudo kill -9 $(pgrep '{}|{}|{}')".format(spec_names[5],spec_names[6],spec_names[8])
        os.system(DEF_KILL_CMD)