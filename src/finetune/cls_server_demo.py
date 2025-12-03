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

# 客户端向服务端发送地震数据，服务端处理后返回结果

# server.py 这是服务端代码文件

# Flask是一个轻量级的Python Web框架，非常适合快速构建API接口
from flask import Flask, request, jsonify
import obspy
import tempfile
import os
import obspyh5

app = Flask(__name__)

import sys
sys.path.append("/home/disk/wd_black/wzm/Sustech_Pulse/src/finetune/wzm/dpk_utils")
from dpk_utils import DiTing_EQDet_PhasePick_predict_fastV2, DiTing_EQDet_PhasePick_predict_fastV2_multievents

# from diting_lsm_model import load_DiTing100M_preveiw, DiTing_EQDet_PhasePick_predict , DiTing_EQDet_PhasePick_predict_fastV2_multievents
from pathlib import Path
import obspy
import numpy as np
import numpy.ma as ma
import torch
sys.path.append("/home/disk/wd_black/wzm/Sustech_Pulse/model/DiTing/ditingbench_preview")
from diting_lsm_model import load_DiTing100M_preveiw #, DiTing_EQDet_PhasePick_predict , DiTing_EQDet_PhasePick_predict_fastV2_multievents , DiTing_EQDet_PhasePick_predict_fastV2


from diting_all import DiTing_EQDet_phase_TTA_predict,DiTing_EQDet_Classify_predict, DiTing_EQDet_magnitude_predict, DiTing_EQDet_magnitude_predict_4_7 , DiTing_EQDet_phase_TTA_predict_2
# 可以改成加载自己模型
# 加载1.2B模型
dpk_model_path = '/home/disk/wd_black/wzm/Sustech_Pulse/model/DiTing/DiTing0.1B-preview/DiTing0.1B-preview.pth'
dpk_device = 'cuda:0'
dpk_model = load_DiTing100M_preveiw( weight_path=dpk_model_path, device=dpk_device)

# 路由装饰器，用于定义如何处理特定的 HTTP 请求
# 当请求 POST /diting_example 到达时，检查请求路径是否匹配 /diting_example，请求方法是否为 POST
# 调用与该路由关联的（下方紧邻的）视图函数
@app.route("/diting_dpk", methods=["POST"])

# 视图函数，由HTTP请求触发，自动注入 request 对象
def diting_dpk():
    # 1. 获取 HDF5 文件
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    # 自动完成解包，具体包括：协议解析、二进制分离、元数据提取
    upload = request.files['file']

    if upload.filename == '':
        return jsonify({"error": "no selected file"}), 400

    # 2. 保存上传的文件到临时目录
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, upload.filename)
    upload.save(tmp_path)

    # 3. 读取 HDF5 文件 （其中'stream'的key对应波形数据，后面可以根据需要再调整）
    # 如果直接读取h5为stream需要安装obspyh5（https://github.com/trichter/obspyh5)
    st = obspy.read(tmp_path, format="MSEED", stream=True)
    if st[0].stats.sampling_rate != 100.0:
        st.resample(100.0)
    os.remove(tmp_path)
    # 4. 推理并保存结果
    results = {}
    station_list = []
    for tr in st:
        station_list.append(tr.stats.station)
    station_list = list(set(station_list))

    for sta in station_list:
        results[sta] = {}
        results[sta]['P'] = []
        results[sta]['S'] = []

        # input_st = st.select(station=sta)
        # # 推理并且返回结果
        # final_P,final_S = DiTing_EQDet_phase_TTA_predict_2(input_st, dpk_device, dpk_model )
        # # results[sta]['P'].append(str(final_P))
        # # results[sta]['S'].append(str(final_S))
        # results[sta]['P'].append(str(input_st[0].stats.starttime + final_P))
        # results[sta]['S'].append(str(input_st[0].stats.starttime + final_S))

        # # events = DiTing_EQDet_PhasePick_predict_fastV2(input_st, dpk_device, dpk_model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.30, batch_size=100)
        # # for t_event in events:
        # #     results[sta]['P'].append(str(input_st[0].stats.starttime + t_event[1][0][0]/100.0))
        # #     results[sta]['S'].append(str(input_st[0].stats.starttime + t_event[2][0][0]/100.0))
        input_st = st.select(station=sta)

        #对长的波形做multi event repeat (或者事件比较多，就走多事件的分支)
        if input_st[0].stats.endtime - input_st[0].stats.starttime > 8*60:
            input_st = input_st.filter('bandpass',freqmin=0.005, freqmax=50.0) 
            events = DiTing_EQDet_PhasePick_predict_fastV2_multievents(input_st, dpk_device, dpk_model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.20, batch_size=100, max_repeat=7)
        else:
            input_st = input_st.detrend('demean')
            events = DiTing_EQDet_PhasePick_predict_fastV2(input_st, dpk_device, dpk_model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.30, batch_size=100, return_confidence=False)
          
        for t_event in events:
            try:
                results[sta]['P'].append(str(input_st[0].stats.starttime + t_event[1][0][0]/100.0))
                # results[sta]['P_for_compare'].append( t_event[1][0][0]/100.0 )
            except:
                pass
            try:
                results[sta]['S'].append(str(input_st[0].stats.starttime + t_event[2][0][0]/100.0))
                # results[sta]['S_for_compare'].append( t_event[2][0][0]/100.0 )
            except:
                    pass
        #  # events = DiTing_EQDet_PhasePick_predict_fastV2(input_st, dpk_device, dpk_model, window_length=10000, step_size=1000, p_th=0.1, s_th=0.1, det_th=0.50, batch_size=100, return_confidence=False)
        #     final_P,final_S = DiTing_EQDet_phase_TTA_predict_2(input_st, dpk_device, dpk_model )
        #     # results[sta]['P'].append(str(final_P))
        #     # results[sta]['S'].append(str(final_S))
        #     results[sta]['P'].append(str(input_st[0].stats.starttime + final_P))
        #     # results[sta]['P_for_compare'].append( final_P )
        #     results[sta]['S'].append(str(input_st[0].stats.starttime + final_S))
        #     # results[sta]['S_for_compare'].append( final_S )
    # 5. 返回 JSON
    return jsonify(results), 200


@app.route("/diting_cls", methods=["POST"])

# 视图函数，由HTTP请求触发，自动注入 request 对象
def diting_cls():
    # 1. 获取 HDF5 文件
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    # 自动完成解包，具体包括：协议解析、二进制分离、元数据提取
    upload = request.files['file']

    if upload.filename == '':
        return jsonify({"error": "no selected file"}), 400

    # 2. 保存上传的文件到临时目录
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, upload.filename)
    upload.save(tmp_path)

    # 3. 读取 HDF5 文件 （其中'stream'的key对应波形数据，后面可以根据需要再调整）
    # 如果直接读取h5为stream需要安装obspyh5（https://github.com/trichter/obspyh5)
    st = obspy.read(tmp_path, format="MSEED", stream=True)
    if st[0].stats.sampling_rate != 100.0:
        st.resample(100.0)
    os.remove(tmp_path)
    # 4. 推理并保存结果
    results = {}
    station_list = []
    for tr in st:
        station_list.append(tr.stats.station)
    station_list = list(set(station_list))

    for sta in station_list:
        results[sta] = 0
        input_st = st.select(station=sta)
        # 推理并且返回结果
        output_class = DiTing_EQDet_Classify_predict(input_st)
        results[sta] =  str(output_class)
        print()
    # 5. 返回 JSON
    return jsonify(results), 200


@app.route("/diting_mag", methods=["POST"])

# 视图函数，由HTTP请求触发，自动注入 request 对象
def diting_mag():
    # 1. 获取 HDF5 文件
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    # 自动完成解包，具体包括：协议解析、二进制分离、元数据提取
    upload = request.files['file']

    if upload.filename == '':
        return jsonify({"error": "no selected file"}), 400

    # 2. 保存上传的文件到临时目录
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, upload.filename)
    upload.save(tmp_path)

    # 3. 读取 HDF5 文件 （其中'stream'的key对应波形数据，后面可以根据需要再调整）
    # 如果直接读取h5为stream需要安装obspyh5（https://github.com/trichter/obspyh5)
    st = obspy.read(tmp_path, format="MSEED", stream=True)
    if st[0].stats.sampling_rate != 100.0:
        st.resample(100.0)
    os.remove(tmp_path)
    # 4. 推理并保存结果
    results = {}
    station_list = []
    for tr in st:
        station_list.append(tr.stats.station)
    station_list = list(set(station_list))

    for sta in station_list:
        results[sta] = 0

        input_st = st.select(station=sta)
        # 推理并且返回结果
        output_mag = DiTing_EQDet_magnitude_predict(input_st)
        #后处理策略
        if output_mag >= 3 and output_mag  < 5:
            output_mag  += 0.6
        if output_mag  >= 2 and output_mag  < 3:
            output_mag  += 0.6

        results[sta] = str(output_mag)
    # 5. 返回 JSON
    return jsonify(results), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8989, debug=False)