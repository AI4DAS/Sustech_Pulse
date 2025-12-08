import numpy as np
import requests
import pandas as pd
import obspy
import numpy.ma as ma
from diting import DiTing_EQDet_magnitude_predict
# server_ip为运行服务的机器ip,
# server_port 为 server_conf.yml 中配置的端口号
server_ip = "127.0.0.1"
server_port = "10090"

# x 应该具有形状 (n,3,10000), 表示 n 条 3 分量波形, 每条波形有 10000 个点
# 第一分量为 Z, 第二分量为 N, 第三分量为 E.
answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/T2.an.txt",sep='\s+' ,names=['filename','magnitude'])

point = 0
with open("emg_output.txt","w") as f:
    f.write("i diff")
for i in range(1,200):
    file_name  = answer.iloc[i,0]
    st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/'+file_name[9:])


    output = DiTing_EQDet_magnitude_predict(st)
    # # 在向 api 提交数据的时候,需要将数组转换成 list,
    # # 并封装到字典里, 字典的 key 为 'array_data'
    # data = {'array_data': np.nan_to_num(st, nan=0.0, posinf=1e10, neginf=-1e10).tolist()}

    # # 通过对应的 api 接口调用 DiTing 模型进行计算
    # # 接口统一为 inference, 根据配置文件确定不同的任务类型
    #output = requests.post(f'http://{server_ip}:{server_port}/inference/', json=data)

    # 打印结果
    print(output, answer.iloc[i,1])
    diff = abs(output - answer.iloc[i,1])
    print(diff)
    if diff <= 0.2:
        point += 1
    elif diff <= 0.6:
        point += 1.5 - 2.5*diff
    with open("emg_output.txt", "a") as f:
        f.write(f"{i} {diff:.2f}\n")
print(point) 