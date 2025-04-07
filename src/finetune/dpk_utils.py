import torch
import numpy as np
from obspy.signal.trigger import trigger_onset

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    modified from https://github.com/smousavi05/EQTransformer
    Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind



def postprocesser_ev_center(yh1, yh2, yh3, det_th=0.3, p_th=0.3, p_mpd=10, s_th=0.3, s_mpd=10, ev_tolerance = 100, p_tolerance = 500):

    """ 
    modified from https://github.com/smousavi05/EQTransformer
    Postprocessing to detection and phase picking
    """         
             
    detection = trigger_onset(yh1, det_th, det_th)
    pp_arr = _detect_peaks(yh2, mph=p_th, mpd=p_mpd)
    ss_arr = _detect_peaks(yh3, mph=s_th, mpd=s_mpd)
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = list()

    # P
    if len(pp_arr) > 0:
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob]})                 
    # S         
    if len(ss_arr) > 0:            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob]})             
            
    if len(detection) > 0:
        # merge close detections
        for ev in range(1,len(detection)):
            if detection[ev][0] - detection[ev-1][1] < ev_tolerance:
                detection[ev-1][1] = detection[ev][1]
                detection[ev][0] = -1
                detection[ev][1] = -1

        for ev in range(len(detection)):                                 
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
            EVENTS.update({ detection[ev][0] : [D_prob, detection[ev][1]]})            
    
    # matching the detection and picks
    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][1]

        if int(ed-bg) >= ev_tolerance:
            candidate_Ps = list()
            for Ps, P_val in P_PICKS.items():
                if Ps > bg - p_tolerance and Ps < bg + p_tolerance:
                    candidate_Ps.append([Ps, P_val[0]])
            
            candidate_Ss = list()
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.append([Ss, S_val[0]])
            
            if len(candidate_Ps) == 0:
                continue

            if len(candidate_Ps) != 0 or len(candidate_Ss) != 0:
                if len(candidate_Ps) == 0:
                    candidate_Ps.append([np.nan, np.nan])
                else:
                    #keep the first arrival
                    min_arrival = None
                    for Ps, P_val in candidate_Ps:
                        if min_arrival is None:
                            min_arrival = Ps
                        if Ps < min_arrival:
                            min_arrival = Ps
                    candidate_Ps = [[Ps, P_val] for Ps, P_val in candidate_Ps if Ps == min_arrival]
                    candidate_Ps = [candidate_Ps[0]]

                if len(candidate_Ss) == 0:
                    candidate_Ss.append([np.nan, np.nan])
                else:
                    #keep the first arrival
                    min_arrival = None
                    for Ss, S_val in candidate_Ss:
                        if min_arrival is None:
                            min_arrival = Ss
                        if Ss < min_arrival:
                            min_arrival = Ss
                    candidate_Ss = [[Ss, S_val] for Ss, S_val in candidate_Ss if Ss == min_arrival]
                    candidate_Ss = [candidate_Ss[0]]

                matches.append([bg, candidate_Ps, candidate_Ss, ed])
    return matches


def detect_peaks_post_process(yh1, yh2, p_th=0.3, p_mpd=10, s_th=0.3, s_mpd=10):
    
    pp_arr = _detect_peaks(yh1, mph=p_th, mpd=p_mpd)

    ss_arr = _detect_peaks(yh2, mph=s_th, mpd=s_mpd)

    P_PICKS = {}
    S_PICKS = {}

    # P
    if len(pp_arr) > 0:
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
            if pauto: 
                P_prob = np.round(yh1[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob]})                 
    # S         
    if len(ss_arr) > 0:            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
            if sauto: 
                S_prob = np.round(yh2[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob]})

    return P_PICKS, S_PICKS

def DiTing_EQDet_PhasePick_predict(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50):
    data_len = stream[0].data.shape[0]
    tmp_waveform = np.zeros([data_len, 3])
    # 头段里面的，后面可以优化
    try:
        tmp_waveform[:,0] = stream.select(channel='*Z')[0].data
    except:
        pass
    try:
        tmp_waveform[:,1] = stream.select(channel='*[N1]')[0].data
    except:
        pass
    try:
        tmp_waveform[:,2] = stream.select(channel='*[E2]')[0].data
    except:
        pass
    
    if data_len < window_length:
        num_windows = 1
        count = np.zeros((1,3,10000))
        confidence = np.zeros((1,3,10000))

    else:
        num_windows = (data_len - window_length) // step_size + 1
        # print(num_windows)
        count = np.zeros((1,3,tmp_waveform.shape[0]))
        confidence = np.zeros((1,3,tmp_waveform.shape[0]))

    for i in range(num_windows):
        if i % 100 == 0:
            print(f"Processing window {i+1}/{num_windows}")
        start = i * step_size
        end = start + window_length
        
        count[:,:,start:end] += 1
        
        window = tmp_waveform[start:end, :].copy()

        # Perform operations on the windowed data
        for chdx in range(3):
            window[:,chdx] -= np.mean(window[:,chdx])
            window[:,chdx] /= np.std(window[:,chdx])

        # Fill empty window with zeros
        if window.shape[0] < window_length:
            padding = np.zeros((window_length - window.shape[0], window.shape[1]))
            window = np.vstack((window, padding))
        
        window_tensor = torch.from_numpy(window)[None, :]
        window_tensor = window_tensor.permute(0, 2, 1)
        window_tensor = window_tensor.to(device).float()

        output = model(window_tensor)
        output =  torch.concat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
        output_np = output.cpu().detach().numpy()
        
        confidence[:,:,start:end] = confidence[:,:,start:end] + output_np
        
    confidence = confidence / count
    # 使用所有窗口confidence均值预测的结果

    events = postprocesser_ev_center(
        yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th,det_th=det_th)

    if len(events) == 0:
        events = [[np.nan, [[np.nan, np.nan]],[[np.nan, np.nan]]]]

    return events, confidence


def DiTing_EQDet_PhasePick_predict_for_mulit_events(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50, max_repeat=3):
    data_len = stream[0].data.shape[0]
    tmp_waveform = np.zeros([data_len, 3])
    # 头段里面的，后面可以优化
    try:
        tmp_waveform[:,0] = stream.select(channel='*Z')[0].data
    except:
        pass
    try:
        tmp_waveform[:,1] = stream.select(channel='*[N1]')[0].data
    except:
        pass
    try:
        tmp_waveform[:,2] = stream.select(channel='*[E2]')[0].data
    except:
        pass

    events = []

    for _ in range(max_repeat):
        if data_len < window_length:
            num_windows = 1
            count = np.zeros((1,3,10000))
            confidence = np.zeros((1,3,10000))
        else:
            num_windows = (data_len - window_length) // step_size + 1
            # print(num_windows)
            count = np.zeros((1,3,tmp_waveform.shape[0]))
            confidence = np.zeros((1,3,tmp_waveform.shape[0]))

        for i in range(num_windows):
            if i % 100 == 0:
                print(f"Processing window {i+1}/{num_windows}")
            start = i * step_size
            end = start + window_length
            
            count[:,:,start:end] += 1
            
            window = tmp_waveform[start:end, :].copy()

            # Perform operations on the windowed data
            for chdx in range(3):
                window[:,chdx] -= np.mean(window[:,chdx])
                window[:,chdx] /= np.std(window[:,chdx])

            # Fill empty window with zeros
            if window.shape[0] < window_length:
                padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                window = np.vstack((window, padding))
            
            window_tensor = torch.from_numpy(window)[None, :]
            window_tensor = window_tensor.permute(0, 2, 1)
            window_tensor = window_tensor.to(device).float()

            output = model(window_tensor)
            output =  torch.concat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
            output_np = output.cpu().detach().numpy()
            
            confidence[:,:,start:end] = confidence[:,:,start:end] + output_np
            
        confidence = confidence / count
        # 使用所有窗口confidence均值预测的结果

        cur_events = postprocesser_ev_center(
            yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th,det_th=det_th)

        if len(cur_events) == 0 and len(events) == 0:
            events = [[np.nan, [[np.nan, np.nan]],[[np.nan, np.nan]]]]
            return events, confidence
        elif len(cur_events) == 0:
            return events, confidence
        else:
            events += cur_events
            for t_event in cur_events:
                tmp_waveform[t_event[0]:t_event[0] + int((t_event[3]-t_event[0])*0.8),:] = 0
                

    return events, confidence

def DiTing_EQDet_PhasePick_predict_fastV1(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50, batch_size=500):
    data_len = stream[0].data.shape[0]
    
    # Pre-allocate and normalize waveforms
    tmp_waveform = np.zeros([data_len, 3])
    try:
        tmp_waveform[:,0] = stream.select(channel='*Z')[0].data
    except:
        pass
    try:
        tmp_waveform[:,1] = stream.select(channel='*[N1]')[0].data
    except:
        pass
    try:
        tmp_waveform[:,2] = stream.select(channel='*[E2]')[0].data
    except:
        pass
    
    # Normalize the entire waveform to speed up processing
    for chdx in range(3):
        tmp_waveform[:, chdx] -= np.mean(tmp_waveform[:, chdx])
        tmp_waveform[:, chdx] /= np.std(tmp_waveform[:, chdx])

    if data_len < window_length:
        num_windows = 1
        count = np.zeros((1, 3, data_len))
        confidence = np.zeros((1, 3, data_len))
    else:
        num_windows = (data_len - window_length) // step_size + 1
        count = np.zeros((1, 3, data_len))
        confidence = np.zeros((1, 3, data_len))

    # Loop to accumulate windows
    for i in range(0, num_windows, batch_size):
        end_idx = min(i + batch_size, num_windows)
        windows_batch = []

        # Accumulate windows in a batch
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            window = tmp_waveform[start:end, :].copy()

            # Fill empty window with zeros (if needed)
            if window.shape[0] < window_length:
                padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                window = np.vstack((window, padding))
                
            windows_batch.append(window)

        # Convert batch to tensor and pass through model
        windows_tensor = torch.tensor(np.array(windows_batch), dtype=torch.float32).to(device)
        windows_tensor = windows_tensor.permute(0, 2, 1)  # Change shape to (batch, channels, samples)
        with torch.no_grad():
            output = model(windows_tensor)
        output_combined = torch.cat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
        output_np = output_combined.cpu().detach().numpy()

        # Update count and confidence arrays
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            confidence[:, :, start:end] += output_np[j - i]
            count[:, :, start:end] += 1
    
    # Final confidence calculation
    confidence = confidence / count
    events = postprocesser_ev_center(
        yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th, det_th=det_th)

    if len(events) == 0:
        events = [[np.nan, [[np.nan, np.nan]], [[np.nan, np.nan]]]]
    
    return events, confidence

def DiTing_EQDet_PhasePick_predict_fastV2(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50, batch_size=100, return_confidence=False):
    data_len = stream[0].data.shape[0]
    
    # Pre-allocate and normalize waveforms
    tmp_waveform = np.zeros([data_len, 3])
    try:
        tmp_waveform[:,0] = stream.select(channel='*Z')[0].data
    except:
        pass
    try:
        tmp_waveform[:,1] = stream.select(channel='*[N1]')[0].data
    except:
        pass
    try:
        tmp_waveform[:,2] = stream.select(channel='*[E2]')[0].data
    except:
        pass
    
    if data_len < window_length:
        num_windows = 1
        tmp_waveform_pad = np.zeros([window_length, 3])
        tmp_waveform_pad[:data_len,:] = tmp_waveform[:,:]
        tmp_waveform = tmp_waveform_pad
        count = np.zeros((1, 3, window_length))
        confidence = np.zeros((1, 3, window_length))
    else:
        num_windows = (data_len - window_length) // step_size + 1
        count = np.zeros((1, 3, data_len))
        confidence = np.zeros((1, 3, data_len))

    # Loop to accumulate windows
    for i in range(0, num_windows, batch_size):
        end_idx = min(i + batch_size, num_windows)
        windows_batch = []

        # Accumulate windows in a batch
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            window = tmp_waveform[start:end, :].copy()
            # Normalize the entire waveform to speed up processing
            for chdx in range(3):
                window[:, chdx] -= np.mean(window[:, chdx])
                norm_factor = np.std(window[:, chdx])
                if norm_factor == 0:
                    pass
                else:
                    window[:, chdx] /= norm_factor
            
            # Fill empty window with zeros (if needed)
            if window.shape[0] < window_length:
                padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                window = np.vstack((window, padding))
                
            windows_batch.append(window)

        # Convert batch to tensor and pass through model
        windows_tensor = torch.tensor(np.array(windows_batch), dtype=torch.float32).to(device)
        windows_tensor = windows_tensor.permute(0, 2, 1)  # Change shape to (batch, channels, samples)
        with torch.no_grad():
            output = model(windows_tensor)
        output_combined = torch.cat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
        output_np = output_combined.cpu().detach().numpy()

        # Update count and confidence arrays
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            confidence[:, :, start:end] += output_np[j - i]
            count[:, :, start:end] += 1
    
    # Final confidence calculation
    confidence = confidence / count
    events = postprocesser_ev_center(
        yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th, det_th=det_th)

    if len(events) == 0:
        events = []
    
    if return_confidence:
        return events, confidence
    else:
        return events


def DiTing_EQDet_PhasePick_predict_fastV2_multievents(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50, batch_size=500, max_repeat=3):
    data_len = stream[0].data.shape[0]
    
    # Pre-allocate and normalize waveforms
    tmp_waveform = np.zeros([data_len, 3])
    try:
        tmp_waveform[:,0] = stream.select(channel='*Z')[0].data
    except:
        pass
    try:
        tmp_waveform[:,1] = stream.select(channel='*[N1]')[0].data
    except:
        pass
    try:
        tmp_waveform[:,2] = stream.select(channel='*[E2]')[0].data
    except:
        pass
    
    events = []
    if data_len < window_length:
        merged_confidence = np.zeros((1, 3, window_length))
    else:
        merged_confidence = np.zeros((1, 3, data_len))
    
    for _ in range(max_repeat):
        print('On {}'.format(_))
        if data_len < window_length:
            num_windows = 1
            tmp_waveform_pad = np.zeros([window_length, 3])
            tmp_waveform_pad[:data_len,:] = tmp_waveform[:data_len,:]
            tmp_waveform = tmp_waveform_pad
            count = np.zeros((1, 3, window_length))
            confidence = np.zeros((1, 3, window_length))
        else:
            num_windows = (data_len - window_length) // step_size + 1
            count = np.zeros((1, 3, data_len))
            confidence = np.zeros((1, 3, data_len))

        # Loop to accumulate windows
        for i in range(0, num_windows, batch_size):
            end_idx = min(i + batch_size, num_windows)
            windows_batch = []

            # Accumulate windows in a batch
            for j in range(i, end_idx):
                start = j * step_size
                end = start + window_length
                window = tmp_waveform[start:end, :].copy()
                # Normalize the entire waveform to speed up processing
                for chdx in range(3):
                    window[:, chdx] -= np.mean(window[:, chdx])
                    norm_factor = np.std(window[:, chdx])
                    if norm_factor == 0:
                        pass
                    else:
                        window[:, chdx] /= norm_factor
                
                # Fill empty window with zeros (if needed)
                if window.shape[0] < window_length:
                    padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                    window = np.vstack((window, padding))
                    
                windows_batch.append(window)

            # Convert batch to tensor and pass through model
            windows_tensor = torch.tensor(np.array(windows_batch), dtype=torch.float32).to(device)
            windows_tensor = windows_tensor.permute(0, 2, 1)  # Change shape to (batch, channels, samples)
            with torch.no_grad():
                output = model(windows_tensor)
            output_combined = torch.cat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
            output_np = output_combined.cpu().detach().numpy()

            # Update count and confidence arrays
            for j in range(i, end_idx):
                start = j * step_size
                end = start + window_length
                confidence[:, :, start:end] += output_np[j - i]
                count[:, :, start:end] += 1
        
        # Final confidence calculation
        confidence = confidence / count

        #merged_confidence = np.maximum(merged_confidence, confidence)

        cur_events = postprocesser_ev_center(
            yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th, det_th=det_th)

        if len(cur_events) == 0 and len(events) == 0:
            events = []
            return events
        elif len(cur_events) == 0:
            return events
        else:
            events += cur_events
            for t_event in cur_events:
                P = t_event[1][0][0]
                S = t_event[2][0][0]

                if np.isnan(P):
                    mask_start = S - 300
                else:
                    mask_start = P - 300
                mask_start = np.max([0, mask_start])

                if np.isnan(S):
                    mask_end = P + 300
                elif np.isnan(P):
                    mask_end = S + 300
                else:
                    mask_end = int(P + (S - P)*1.2)

                mask_end = np.min([len(tmp_waveform[:,0]), mask_end])
                
                tmp_waveform[mask_start:mask_end ,:] = 0

    return events


def DiTing_EQDet_PhasePick_predict_fastV2_array(tmp_waveform, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.50, batch_size=500):
    data_len = tmp_waveform.shape[0]
    
    if data_len < window_length:
        num_windows = 1
        tmp_waveform_pad = np.zeros([window_length, 3])
        tmp_waveform_pad[:data_len,:] = tmp_waveform[:,:]
        tmp_waveform = tmp_waveform_pad
        count = np.zeros((1, 3, window_length))
        confidence = np.zeros((1, 3, window_length))
    else:
        num_windows = (data_len - window_length) // step_size + 1
        count = np.zeros((1, 3, data_len))
        confidence = np.zeros((1, 3, data_len))

    # Loop to accumulate windows
    for i in range(0, num_windows, batch_size):
        end_idx = min(i + batch_size, num_windows)
        windows_batch = []

        # Accumulate windows in a batch
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            window = tmp_waveform[start:end, :].copy()
            # Normalize the entire waveform to speed up processing
            for chdx in range(3):
                window[:, chdx] -= np.mean(window[:, chdx])
                norm_factor = np.std(window[:, chdx])
                if norm_factor == 0:
                    pass
                else:
                    window[:, chdx] /= norm_factor
            
            # Fill empty window with zeros (if needed)
            if window.shape[0] < window_length:
                padding = np.zeros((window_length - window.shape[0], window.shape[1]))
                window = np.vstack((window, padding))
                
            windows_batch.append(window)

        # Convert batch to tensor and pass through model
        windows_tensor = torch.tensor(np.array(windows_batch), dtype=torch.float32).to(device)
        windows_tensor = windows_tensor.permute(0, 2, 1)  # Change shape to (batch, channels, samples)
        with torch.no_grad():
            output = model(windows_tensor)
        output_combined = torch.cat((output['det'].unsqueeze(1), output['ppk'].unsqueeze(1), output['spk'].unsqueeze(1)), dim=1)
        output_np = output_combined.cpu().detach().numpy()

        # Update count and confidence arrays
        for j in range(i, end_idx):
            start = j * step_size
            end = start + window_length
            confidence[:, :, start:end] += output_np[j - i]
            count[:, :, start:end] += 1
    
    # Final confidence calculation
    confidence = confidence / count
    events = postprocesser_ev_center(
        yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], p_th=p_th, s_th=s_th, det_th=det_th)

    #if len(events) == 0:
    #    events = [[np.nan, [[np.nan, np.nan]], [[np.nan, np.nan]]]]
    
    return events


def DiTing_EQDet_PhasePick_predict_fast_for_RealTimeStream(total_stream, device, model,
                                                             window_length=10000, step_size=3000,
                                                             p_th=0.1, s_th=0.1, det_th=0.50,
                                                             batch_size=500):
    sta_phase_dict = dict()
    net_sta_list = list(set(trace.stats.network + '.' + trace.stats.station for trace in total_stream.traces))
    
    total_stream.merge(fill_value=0)
    total_stream.detrend('demean')
        # 存储各台站预处理后的数据及其长度
    station_data = dict()
    for net_sta in net_sta_list:
        net, sta = net_sta.split('.')
        sta_stream = total_stream.select(network=net, station=sta)
        if sta_stream[0].stats.sampling_rate != 100:
            sta_stream.resample(100.0)
        # 强制对齐起始终止
        sta_stream.trim(starttime=sta_stream[0].stats.starttime, endtime=sta_stream[0].stats.endtime,pad=True, fill_value=0)

        # 构造 waveform，注意异常处理
        data_len = sta_stream[0].data.shape[0]
        waveform = np.zeros((data_len, 3))
        waveform[:, 0] = sta_stream.select(channel='*Z')[0].data
        try:
            waveform[:, 1] = sta_stream.select(channel='*[N1]')[0].data
        except:
            pass
        try:
            waveform[:, 2] = sta_stream.select(channel='*[E2]')[0].data
        except:
            pass
        
        # 如果数据不足，则填零补全到 window_length
        if data_len < window_length:
            tmp = np.zeros((window_length, 3))
            tmp[:data_len, :] = waveform
            waveform = tmp
        
        station_data[net_sta] = {'waveform': waveform, 'data_len': waveform.shape[0]}
    
    # 构建所有台站的滑动窗口列表和映射信息
    windows_info = []  # 每个元素记录 (net_sta, start_idx, window)
    for net_sta, data in station_data.items():
        waveform = data['waveform']
        data_len = data['data_len']
        if data_len < window_length:
            windows_info.append((net_sta, 0, waveform))
        else:
            num_windows = (data_len - window_length) // step_size + 1
            for j in range(num_windows):
                start = j * step_size
                end = start + window_length
                window = waveform[start:end, :].copy()
                # 对每个通道归一化
                for ch in range(3):
                    window[:, ch] -= np.mean(window[:, ch])
                    std_val = np.std(window[:, ch])
                    if std_val != 0:
                        window[:, ch] /= std_val
                windows_info.append((net_sta, start, window))
    
    # 初始化每个台站的累计计数与置信度数组
    station_count = {}
    station_confidence = {}
    for net_sta, data in station_data.items():
        L = data['data_len']
        station_count[net_sta] = np.zeros((1, 3, L))
        station_confidence[net_sta] = np.zeros((1, 3, L))
    
    # 批处理所有窗口
    for i in range(0, len(windows_info), batch_size):
        batch_windows = windows_info[i:i+batch_size]
        batch_array = []
        mapping = []  # 记录每个窗口对应的 (net_sta, start_idx)
        for (net_sta, start_idx, window) in batch_windows:
            batch_array.append(window)
            mapping.append((net_sta, start_idx))
        
        # 转为张量并调整 shape
        windows_tensor = torch.tensor(np.array(batch_array), dtype=torch.float32).to(device)
        windows_tensor = windows_tensor.permute(0, 2, 1)  # (batch, channels, samples)
        with torch.no_grad():
            output = model(windows_tensor)
        # 拼接各个通道的输出
        output_combined = torch.cat((output['det'].unsqueeze(1),
                                     output['ppk'].unsqueeze(1),
                                     output['spk'].unsqueeze(1)), dim=1)
        output_np = output_combined.cpu().detach().numpy()
        
        # 根据 mapping 信息，更新每个台站对应位置的结果
        for idx, (net_sta, start_idx) in enumerate(mapping):
            end_idx = start_idx + window_length
            # 注意若 end_idx 超出数据长度，则需要截断
            L = station_data[net_sta]['data_len']
            if end_idx > L:
                end_idx = L
            station_confidence[net_sta][:, :, start_idx:end_idx] += output_np[idx][:, :end_idx-start_idx]
            station_count[net_sta][:, :, start_idx:end_idx] += 1
    
    # 后处理：计算置信度，并调用后处理函数得到事件列表
    for net_sta in net_sta_list:
        count = station_count[net_sta]
        confidence = station_confidence[net_sta] / count
        events = postprocesser_ev_center(
            yh1=confidence[0, 0, :],
            yh2=confidence[0, 1, :],
            yh3=confidence[0, 2, :],
            p_th=p_th, s_th=s_th, det_th=det_th)
        if len(events) == 0:
            #events = [[np.nan, [[np.nan, np.nan]], [[np.nan, np.nan]]]]
            continue

        sta_phase_dict[net_sta] = events

    return sta_phase_dict
