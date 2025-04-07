
'''
API接口（Application Programming Interface）
包括客户端 (Client Side)和服务端 (Server Side)

客户端             服务端
    │               │
    │  HTTP请求      │
    │──────────────>│
    │               │ 处理请求
    │               │ 访问数据库
    │               │ 执行业务逻辑
    │  HTTP响应     │
    │<──────────────│
    │               │
'''

# client.py 这是客户端代码文件，向服务端发送地震数据

import requests  # 导入requests库，用于发送HTTP请求
import pandas as pd
import os
def diting_dpk(url):
    """
    向服务端发送HDF5文件的函数
    参数:
        h5_filepath (str): 要上传的HDF5文件路径
    返回:
        dict: 服务端返回的JSON响应数据，出错时返回None
    """
        
    answer = pd.read_csv("/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-an/T1.an",sep='\s+' ,usecols=[ 0 , 2 , 4],names=['filename','P' ,'S'])
    predict_list = []
    for i in range(5,7):
        file_name  = answer.iloc[i,0]
        st_filepath = '/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-exam/'+file_name[2:]
        # 文件二进制编码
        # 打包文件准备发送
        files = {
            "file": open(st_filepath, "rb")  # "file"对应服务端接收的参数名
        }
        try:
            # 将HDF5文件内容作为二进制流写入请求体，具体包括 二进制文件 -> 自动生成随机边界符 -> 封装请求体 -> 设置请求头
            # 与 127.0.0.1:5000 建立TCP连接后，在已建立的TCP连接上发送一个方法标识为POST的完整HTTP请求报文到服务端
            # 接收服务端返回HTTP响应
            resp = requests.post(url, files=files)

            # 检查响应状态码，如果请求成功（状态码为 200-399）, 如果请求失败（状态码为 400-599）
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 处理所有requests可能抛出的异常
            print(f"请求失败：{e}")
            return None
        # 返回服务端的JSON响应（自动转换为Python字典）
        if resp:
            print(resp.json())
    return "dpk end"

def diting_mag(url):
    """
    向服务端发送HDF5文件的函数
    参数:
        h5_filepath (str): 要上传的HDF5文件路径
    返回:
        dict: 服务端返回的JSON响应数据，出错时返回None
    """
        
    answer = pd.read_csv("/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-an/T2.an",sep='\s+' ,names=['filename','magnitude'])
    predict_list = []
    for i in range(5):
        file_name  = answer.iloc[i,0]
        #file_name  = os.path.basename(file_name)
        st_filepath = '/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-exam/'+file_name[2:]
        # 文件二进制编码
        # 打包文件准备发送
        files = {
            "file": open(st_filepath, "rb")  # "file"对应服务端接收的参数名
        }
        try:
            # 将HDF5文件内容作为二进制流写入请求体，具体包括 二进制文件 -> 自动生成随机边界符 -> 封装请求体 -> 设置请求头
            # 与 127.0.0.1:5000 建立TCP连接后，在已建立的TCP连接上发送一个方法标识为POST的完整HTTP请求报文到服务端
            # 接收服务端返回HTTP响应
            resp = requests.post(url, files=files)

            # 检查响应状态码，如果请求成功（状态码为 200-399）, 如果请求失败（状态码为 400-599）
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 处理所有requests可能抛出的异常
            print(f"请求失败：{e}")
            return None
        # 返回服务端的JSON响应（自动转换为Python字典）
        if resp:
            print(resp.json())
    return "mag end"

def diting_cls(url):
    """
    向服务端发送HDF5文件的函数
    参数:
        h5_filepath (str): 要上传的HDF5文件路径
    返回:
        dict: 服务端返回的JSON响应数据，出错时返回None
    """
        
    answer = pd.read_csv("/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-an/T3.an",sep='\s+' ,names=['filename','class','name'])
    for i in range(5):
        file_name  = answer.iloc[i,0]
        st_filepath = '/home/disk/wd_black/wzm/Sustech_Pulse/test_data/07-exam/'+file_name[2:]
        # 文件二进制编码
        # 打包文件准备发送
        files = {
            "file": open(st_filepath, "rb")  # "file"对应服务端接收的参数名
        }
        try:
            # 将HDF5文件内容作为二进制流写入请求体，具体包括 二进制文件 -> 自动生成随机边界符 -> 封装请求体 -> 设置请求头
            # 与 127.0.0.1:5000 建立TCP连接后，在已建立的TCP连接上发送一个方法标识为POST的完整HTTP请求报文到服务端
            # 接收服务端返回HTTP响应
            resp = requests.post(url, files=files)

            # 检查响应状态码，如果请求成功（状态码为 200-399）, 如果请求失败（状态码为 400-599）
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 处理所有requests可能抛出的异常
            print(f"请求失败：{e}")
            return None
        # 返回服务端的JSON响应（自动转换为Python字典）
        if resp:
            print(resp.json())
    return "cls end"


if __name__ == "__main__":
    import time
    t1 = time.time()
    ip = "118.145.133.95:8989"
    # ip = "127.0.0.1:8989"

    url = f"http://{ip}/diting_dpk"
    result = diting_dpk(url)
    t2 = time.time()
    print('API time: {:}'.format(t2-t1))
    # 如果有返回结果则打印

    t1 = time.time()
    url = f"http://{ip}/diting_cls"
    result = diting_cls(url)
    t2 = time.time()
    print('API time: {:}'.format(t2-t1))

    t1 = time.time()
    url = f"http://{ip}/diting_mag"
    result = diting_cls(url)
    t2 = time.time()
    print('API time: {:}'.format(t2-t1))