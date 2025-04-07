import numpy as np
import requests
import pandas as pd
import obspy
import numpy.ma as ma
from diting_all import DiTing_EQDet_magnitude_predict ,  DiTing_EQDet_Classify_predict

# answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/answer07/T2.an",sep='\s+' ,names=['filename','magnitude'])

# point = 0
# with open("emg_output.txt","w") as f:
#     f.write("i diff")
# for i in range(1,200):
#     file_name  = answer.iloc[i,0]
#     st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam-data07'+file_name[1:])

#     try:
#         output = DiTing_EQDet_magnitude_predict(st)
#         # 打印结果
#         print(output, answer.iloc[i,1])
#         diff = abs(output - answer.iloc[i,1])
#         print(diff)
#         if diff <= 0.2:
#             point += 1
#         elif diff <= 0.6:
#             point += 1.5 - 2.5*diff
#         with open("emg_output.txt", "a") as f:
#             f.write(f"{i} {output} { answer.iloc[i,1]} {diff:.2f}\n")
#     except ValueError as e:
#         print(e)
# print(point) 

# answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/T2.an.txt",sep='\s+' ,names=['filename','magnitude'])
# point2 = point
# point = 0
# with open("emg_output.txt","w") as f:
#     f.write("i diff")
# for i in range(1,200):
#     file_name  = answer.iloc[i,0]
#     st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/'+file_name[9:])


#     output = DiTing_EQDet_magnitude_predict(st)
#     # # 在向 api 提交数据的时候,需要将数组转换成 list,
#     # # 并封装到字典里, 字典的 key 为 'array_data'
#     # data = {'array_data': np.nan_to_num(st, nan=0.0, posinf=1e10, neginf=-1e10).tolist()}

#     # # 通过对应的 api 接口调用 DiTing 模型进行计算
#     # # 接口统一为 inference, 根据配置文件确定不同的任务类型
#     #output = requests.post(f'http://{server_ip}:{server_port}/inference/', json=data)

#     # 打印结果
#     print(output, answer.iloc[i,1])
#     diff = abs(output - answer.iloc[i,1])
#     print(diff)
#     if diff <= 0.2:
#         point += 1
#     elif diff <= 0.6:
#         point += 1.5 - 2.5*diff
#     with open("emg_output.txt", "a") as f:
#         f.write(f"{i}  {output} { answer.iloc[i,1]}  {diff:.2f}\n")




answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/answer07/T3.an",sep='\s+' ,usecols=[ 0 , 1],names=['filename','class'])
score=0
y=[]
for i in range(200):
    file_name  = answer.iloc[i,0]
    try:
        st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam-data07'+file_name[1:])
        class_out = DiTing_EQDet_Classify_predict(st)
        #    print(class_out, answer.iloc[i,1]) 
        if int(class_out)==int(answer.iloc[i,1]):
            if int(class_out) == 1:
                score = score + 0.5
            elif  int(answer.iloc[i,1]) in [1,2,3]:
                score=score+2
            elif int(answer.iloc[i,1]) == 4 :
                score=score+1
        else:
            y.append( [i, class_out , answer.iloc[i,1] ])
    except OSError:
        print("read error")

print("cls score",score)


answer = pd.read_csv("/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/T3.an.txt",sep='\s+' ,names=['filename','class'])
score2 = score

score=0
for i in range(200):
   file_name  = answer.iloc[i,0]
   st = obspy.read('/home/disk/disk02/wzm/Sustech_Pulse/tutorial/exam202506/'+file_name[9:])
   class_out = DiTing_EQDet_Classify_predict(st)
   
   if int(class_out)==int(answer.iloc[i,1]):
        if int(class_out) == 1:
            score = score + 0.5
        elif  int(answer.iloc[i,1]) in [1,2,3]:
            score=score+2
        elif int(answer.iloc[i,1]) == 4 :
            score=score+1
   else:
       y.append([i, class_out , answer.iloc[i,1] ])
       

print("cls score",score , score2)
# print("mag" ,point , point2) 

y = pd.DataFrame(y, columns=['index','output','class'])
print(y)

#cls score 104.5 79.5


