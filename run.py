import time
import subprocess
for i in range(1,11):
    bashCommand = "python train.py --alg=LPD --dataperc=100 --epoch=20 --setup=1 --lr=1e-4 --iterates="+str(i)
    process=subprocess.Popen(bashCommand.split(),cwd='.').wait()
