#-*- coding: UTF-8 -*-
import math
import numpy as np

CNN=np.array([
    [64, 3, 224, 224,3,1,1,1],#Conv_in
    [64, 64, 224, 224,3,1,1,2],
    [128, 64, 112, 112,3,1,1,3],
    [128, 128, 112, 112,3,1,1,4],
    [256, 128, 56, 56,3,1,1,5],
    [256, 256, 56, 56,3,1,1,6],
    [256, 256, 56, 56,3,1,1,7],
    [512, 256, 28, 28,3,1,1,8],
    [512, 512, 28, 28,3,1,1,9],
    [512, 512, 28, 28,3,1,1,10],
    [512, 512, 14, 14,3,1,1,11],
    [512, 512, 14, 14,3,1,1,12],
    [512, 512, 14, 14,3,1,1,13],#FP
    [512, 512, 14, 14, 3, 1, 2, 13],
    [512, 512, 14, 14, 3, 1, 2, 12],
    [512, 512, 14, 14, 3, 1, 2, 11],
    [512, 512, 28, 28, 3, 1, 2, 10],
    [512, 512, 28, 28, 3, 1, 2, 9],
    [256, 512, 28, 28, 3, 1, 2, 8],
    [256, 256, 56, 56,3,1,2,7],
    [256, 256, 56, 56,3,1,2,6],
    [128, 256, 56, 56,3,1,2,5],
    [128, 128, 112, 112,3,1,2,4],
    [64, 128, 112, 112,3,1,2,3],
    [64, 64, 224, 224,3,1,2,2],#BP
    [512, 512, 14, 14, 3, 1, 3, 13],
    [512, 512, 14, 14, 3, 1, 3, 12],
    [512, 512, 14, 14, 3, 1, 3, 11],
    [512, 512, 28, 28, 3, 1, 3, 10],
    [512, 512, 28, 28, 3, 1, 3, 9],
    [512, 256, 28, 28, 3, 1, 3, 8],
    [256, 256, 56, 56, 3, 1, 3, 7],
    [256, 256, 56, 56, 3, 1, 3, 6],
    [128, 256, 56, 56, 3, 1, 3, 5],
    [128, 128, 112, 112, 3, 1, 3, 4],
    [64, 128, 112, 112, 3, 1, 3, 3],
    [64, 64, 224, 224, 3, 1, 3, 2],
    [64, 3, 224, 224,3,1,3,1]#WU
    ],dtype=np.int64) #M,N,C,R,K,S,state,conv_index

# tile cutting parameters
Tm=24
Tn=24
#assert Tm*Tn<NUM_DSP
TrTc_bound=2016
TrTc_in_bound=2486
K_bound=3
TmTn_bound=55


# define a layer parameters
B = 16  # batch size
DMAp=4
tstart=400

def latency(Tr,Tc,R,C,M,N,S,K,M_on,M_on1,state):
    if(N<=4):
        tIFM = tstart + math.ceil(N / DMAp) * ((Tr - 1) * S + K) * ((Tc - 1) * S + K)
    else:
        tIFM=tstart+math.ceil(Tn/DMAp)*((Tr-1)*S+K)*((Tc-1)*S+K)
    tCOMP=Tr*Tc*K*K
    if(state==1):#FP
        tWEI = math.ceil(Tm * Tn / DMAp) * K * K
        tOUT=math.ceil(Tm/DMAp)*Tr*Tc
        tLOAD=max(tIFM,tWEI)
        tPROD1=max(tIFM,tCOMP)
        tPROD2=max(tLOAD,tCOMP)
        tSTORE=max(tCOMP,tOUT)
        lat1=math.ceil(N/Tn-1)*tPROD1+tIFM+tCOMP
        lat2=math.ceil(N/Tn-1)*tPROD1+tIFM+tSTORE
        lat3=(math.ceil(M_on/Tm)*math.ceil(R/Tr)-1)*lat2+lat1+tOUT+ tstart
        latb1=math.ceil(N/Tn-1)*tPROD2+tIFM+tCOMP
        latb2=math.ceil(N/Tn-1)*tPROD2+tIFM+tSTORE
        latb3=math.ceil(M_on/Tm)*math.ceil(R/Tr-1)*lat2+math.ceil(M_on/Tm-1)*latb2+latb1+tOUT+tstart
        latmon0=math.ceil(M/M_on-1)*((B-1)*lat3+latb3)
        tOUT = math.ceil(Tm / DMAp) * Tr * Tc
        tLOAD = max(tIFM, tWEI)
        tPROD1 = max(tIFM, tCOMP)
        tPROD2 = max(tLOAD, tCOMP)
        tSTORE = max(tCOMP, tOUT)
        lat1 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tCOMP
        lat2 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tSTORE
        lat3 = (math.ceil(M_on1 / Tm) * math.ceil(R / Tr) - 1) * lat2 + lat1 + tOUT+ tstart
        latb1 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tCOMP
        latb2 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tSTORE
        latb3 = math.ceil(M_on1 / Tm) * math.ceil(R / Tr - 1) * lat2 + math.ceil(
        M_on1 / Tm - 1) * latb2 + latb1 + tOUT + tstart
        latmon1 = 1 * ((B - 1) * lat3 + latb3)
        lat=latmon0+latmon1
        #print("Tr=",Tr,"cyc=",lat)
    elif (state == 2):  # BP
        tWEI = math.ceil(M_on * Tn / DMAp) * K * K + tstart
        tOUT = math.ceil(Tm / DMAp) * Tr * Tc
        tLOAD = max(tIFM, tWEI)
        tPROD1 = max(tIFM, tCOMP)
        tPROD2 = max(tLOAD, tCOMP)
        tSTORE = max(tCOMP, tOUT)
        lat1 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tCOMP
        lat2 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tSTORE
        lat3 = (math.ceil(M_on / Tm) * math.ceil(R / Tr) - 1) * lat2 + lat1 + tOUT+ tstart
        latb1 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tCOMP
        latb2 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tSTORE
        latb3 = (math.ceil(M_on / Tm )*  math.ceil(R / Tr) - 1) * lat2 + latb1 + tOUT + tstart
        latmon0 = math.ceil(M / M_on - 1) * ((B - 1) * lat3 + latb3)
        tWEI = math.ceil(M_on1 * Tn / DMAp) * K * K + tstart
        tOUT = math.ceil(Tm / DMAp) * Tr * Tc
        tLOAD = max(tIFM, tWEI)
        tPROD1 = max(tIFM, tCOMP)
        tPROD2 = max(tLOAD, tCOMP)
        tSTORE = max(tCOMP, tOUT)
        lat1 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tCOMP
        lat2 = math.ceil(N / Tn - 1) * tPROD1 + tIFM + tSTORE
        lat3 = (math.ceil(M_on1 / Tm) * math.ceil(R / Tr) - 1) * lat2 + lat1 + tOUT+ tstart
        latb1 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tCOMP
        latb2 = math.ceil(N / Tn - 1) * tPROD2 + tIFM + tSTORE
        latb3 =  (math.ceil(M_on1 / Tm )*  math.ceil(R / Tr) - 1) * lat2 + latb1 + tOUT + tstart
        latmon1 = 1 * ((B - 1) * lat3 + latb3)
        lat = latmon0 + latmon1
        #print("Tr=", Tr, "cyc=", lat)
    elif (state == 3):  # WU
        tOUT = math.ceil(Tm * Tn / DMAp) * K * K
        tOFM = math.ceil(Tm / DMAp) * Tr * Tc+tstart
        tLOAD = max(tIFM, tOFM)
        tPROD1 = max(tLOAD, tCOMP)
        tPROD2 = max(tIFM, tCOMP)
        tSTORE = max(tCOMP, tOUT)
        if(Tr<R):
            lat1 = math.ceil(R/ Tr - 1) * tPROD1 + tLOAD + tCOMP
            latb1=math.ceil(R/ Tr - 1) * tPROD1 + tLOAD + tSTORE
            latmon0 = (((B-1)* math.ceil(M_on/Tm)*math.ceil(N/Tn)+1)*lat1+
                       (math.ceil(M_on/Tm)*math.ceil(N/Tn)-1)*latb1+tOUT)*math.ceil(M / M_on - 1)
            latmon1 = (((B - 1) * math.ceil(M_on1 / Tm) * math.ceil(N / Tn) + 1) * lat1 +
                       (math.ceil(M_on1 / Tm) * math.ceil(N / Tn) - 1) * latb1 + tOUT) *1

            lat = latmon0 + latmon1
        else:
            lat1= math.ceil(N/ Tn - 1) * tPROD2 + tLOAD + tCOMP
            latb1=math.ceil(N/ Tn - 1) * (tPROD2+tOUT) + tLOAD + tCOMP+tOUT
            latmon0 = math.ceil(M / M_on - 1)*math.ceil(M_on/Tm)*((B-1)*lat1+latb1)
            latmon1 = 1 * math.ceil(M_on1 / Tm) * ((B - 1) * lat1 + latb1)
            lat=latmon0 + latmon1
    comp = math.ceil(M / Tm) * math.ceil(N / Tn) * math.ceil(R / Tr) * math.ceil(C / Tc) * tCOMP * B
    #print("Tr=", Tr, "cyc=", lat, "comp=",comp)
    return lat



M_on_cnn = []
tr_cnn=[]
lat_cnn=[]
for i in range(0,CNN.shape[0]):
    iter=0
    min_latency=0
    M = CNN[i, 0]  # output channel
    N = CNN[i, 1]  # input channel
    R = CNN[i, 3]  # output row
    C = CNN[i, 2]  # output col
    K = CNN[i, 4]  # weight kernel size
    S = CNN[i, 5]  # stride
    # P = 0  # padding
    state = CNN[i, 6]  # 1FP/2BP/3WU
    Tc = C
    Tc_in = (Tc - 1) * S + K
    multiple = math.floor(M / Tm)
    M_on_updatei = [Tm * j for j in range(1, multiple)]
    if M not in M_on_updatei:
        M_on_updatei.append(CNN[i, 0])
    M_on_updatei = np.sort(M_on_updatei)[::-1]
    #print(M_on_updatei)
    for M_on_i_cur in M_on_updatei:
        M_on_bound=math.floor(TmTn_bound/math.ceil(math.ceil(N/(2*Tn))/(math.floor(K_bound/K)*math.floor(K_bound/K))))*Tm
        #print(M_on_bound)
        if M_on_i_cur <= M_on_bound:
            #print(i,M_on_i_cur)
            break
            #continue
    M_on = M_on_i_cur
    if M_on == M:
        M_on1 = M
    else:
        if M % M_on==0:
            M_on1 = M_on
        else:
            M_on1 = M % M_on
    for tr in range(1, R + 1):
        Tr = tr
        Tr_in = (Tr - 1) * S + K
        if Tr*Tc>TrTc_bound or Tr_in*Tc_in>TrTc_in_bound:
            break
        lat = latency(Tr,Tc,R,C,M,N,S,K,M_on,M_on1,state)
        if iter == 0:
            min_latency = lat
            min_tr = 1
            min_M_on=M_on
        else:
            if lat < min_latency:
                min_tr = tr
                min_latency = lat
                min_M_on = M_on
        iter+=1
        #print(M_on,lat)
    M_on_cnn.append(min_M_on)
    tr_cnn.append(min_tr)
    lat_cnn.append(min_latency)
print(M_on_cnn)
print(tr_cnn)
print(lat_cnn)

print('batch size=',B)
clock=10*1e-9#100MHZ
lat_buffer=[]
throughput_buffer=[]
for i,item in enumerate(CNN):
    if item[6]==1:
        state=' FP: '
    elif item[6]==2:
        state=' BP: '
    elif item[6]==3:
        state=' WU: '
    lat=lat_cnn[i]*clock
    ops=2*item[0]*item[1]*item[2]*item[3]*item[4]*item[4]*B
    throughput=ops/lat*1e-9
    print('Conv',item[7],state,lat,'s',throughput,'GFLOPS')
    lat_buffer.append(lat)
    throughput_buffer.append(ops)
total_lat=np.sum(lat_buffer)
total_throughput=np.sum(throughput_buffer)/total_lat*1e-9
print('all Convs for Vgg',total_lat,'s',total_throughput,'GFLOPS')





