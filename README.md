# CAT_DQN
2CLOS: First CLOS for LLC sensitive benchmarks (omnetpp, xalancbmk), Second CLOS for LLC insensitive benchamrks (lbm)

Use 8 cpu cores (for omnetpp, xalancbmk, lbm\*6 respectively)

State : \[ llc_way(w/11), instructions, cycles, llc_references, llc_misses, CYCLES_L3_MISS \] \* 8 cores

Action : (0)CLOS1-1way , (1)CLOS1+1way , (2)CLOS2-1way , (3)CLOS2+1way , (4)stay

Reward : Sum of IPC ( or Sum of CYCLES_L2_MISS or Sum of 'IPC increment from previous step' ... )

>Modify perf reading code depending on your server.

## Training
`python dqn_train.py 100000`
## Inference
`python dqn_inference.py 10000`

Expected Goal
- increasing the LLC way of CLOS1 to be 10 or 11 => action1
- decreasing the LLC way of CLOS1 to be 1 => action2
