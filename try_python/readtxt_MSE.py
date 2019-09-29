import os
import math

file_name = 'read.txt'
mode = 'r'
cnt_gts, pre_cnts, dm_cnts = [], [], []
with open(file_name, mode) as f:
    # for i in range(30):
    while True:
        tmp = f.readline()
        if not tmp:
            print('{} file ended.'.format(file_name))
            break
            pass
        if tmp[0:5] == 'image':
            cnt_gt = tmp.split('cnt_GT=')[1].split(',')[0]
            cnt_gt = float(cnt_gt)
            cnt_gts.append(cnt_gt)
            pre_cnt = tmp.split('pre_cnt=')[1].split(',')[0]
            pre_cnt = float(pre_cnt)
            pre_cnts.append(pre_cnt)
            dm_cnt = tmp.split('dm_cnt=')[1]
            dm_cnt = float(dm_cnt)
            dm_cnts.append(dm_cnt)
            # print(cnt_gt, pre_cnt, dm_cnt)
m_MSE, m_MAE = 0, 0
alp = 1
m_alpMSE, m_alpMAE = 0, 0
m = len(cnt_gts)
for i in range(m):
    m_MAE += abs(pre_cnts[i]-cnt_gts[i])
    m_MSE += pow(pre_cnts[i]-cnt_gts[i], 2)
    m_alpMAE += abs((pre_cnts[i]+alp*dm_cnts[i])/2 - cnt_gts[i])
    m_alpMSE += pow((pre_cnts[i]+alp*dm_cnts[i])/2 - cnt_gts[i], 2)

MAE = m_MAE / m
MSE = pow(m_MSE / m, 1/2)
alpMAE = m_alpMAE / m
alpMSE = pow(m_alpMSE / m, 1/2)
print(MAE, MSE, alpMAE, alpMSE)

goodMSE, goodMAE = [], []
for alp_int in range(-200, 200):
    alp_float = alp_int / 100
    if 1+alp_float == 0:
        pass
    else:
        m_MSE, m_MAE = 0, 0
        m_alpMSE, m_alpMAE = 0, 0
        for i in range(m):
            m_MAE += abs(pre_cnts[i] - cnt_gts[i])
            m_MSE += pow(pre_cnts[i] - cnt_gts[i], 2)
            m_alpMAE += abs((pre_cnts[i] + alp_float * dm_cnts[i]) / (1+alp_float) - cnt_gts[i])
            m_alpMSE += pow((pre_cnts[i] + alp_float * dm_cnts[i]) / (1+alp_float) - cnt_gts[i], 2)
            # m_alpMAE += abs((alp_float * pre_cnts[i] + dm_cnts[i]) / (1 + alp_float) - cnt_gts[i])
            # m_alpMSE += pow((alp_float * pre_cnts[i] + dm_cnts[i]) / (1 + alp_float) - cnt_gts[i], 2)

        MAE = m_MAE / m
        MSE = pow(m_MSE / m, 1 / 2)
        alpMAE = m_alpMAE / m
        alpMSE = pow(m_alpMSE / m, 1 / 2)
        print('alp_w{}: MAE={}, MSE={}, alpMAE={}, alpMSE={}'.format(alp_float, MAE, MSE, alpMAE, alpMSE))
        if MSE > alpMSE:
            goodMSE.append((alp_float, alpMSE, alpMAE))
        if MAE > alpMAE:
            goodMAE.append((alp_float, alpMSE, alpMAE))
print('goodMSE is: {}'.format(goodMSE))
print('goodMAE is: {}'.format(goodMAE))

