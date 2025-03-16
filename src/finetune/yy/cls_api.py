import numpy as np
import requests
import pandas as pd
import obspy
from diting import DiTing_EQDet_Classify_predict
# server_ip为运行服务的机器ip,
# server_port 为 server_conf.yml 中配置的端口号
server_ip = "127.0.0.1"
server_port = "10080"

# x 应该具有形状 (n,3,10000), 表示 n 条 3 分量波形, 每条波形有 10000 个点
# 第一分量为 Z, 第二分量为 N, 第三分量为 E.
answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/T3.an.txt",sep='\s+' ,names=['filename','class'])
score=0
y=[]
for i in range(200):
   file_name  = answer.iloc[i,0]
   st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/'+file_name[9:])
   class_out = DiTing_EQDet_Classify_predict(st)
   if int(class_out)!=1 and int(answer.iloc[i,1])!=1:
       score = score + 1
   if int(class_out)==int(answer.iloc[i,1]):
        if int(class_out) == 1:
            score = score + 0.5
        else:
            score=score+1
   else:
       y.append([i, class_out , answer.iloc[i,1]])
       

print("cls score",score)
y = pd.DataFrame(y, columns=['index','output','class'])
print(y)
