#plotter_for_journalベース
#ベンチデータを読み込んで、ベストなスコアを出しているものを画面に表示するだけ
import os
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import copy

test_type = 5 # 0: normal, 1: same, 2: without_Curriculum, 3: pow1, 4: original implementation and prey_maintain_duration=5, 5: for journal

TestCases = [['CL_DDPG', 'CL_MADDPG', 'CG_DDPG', 'CG_MADDPG', 'NL_DDPG', 'NL_MADDPG', 'NG_DDPG', 'NG_MADDPG'],
             ['CL_DDPG_same', 'CL_MADDPG_same', 'CG_DDPG_same', 'CG_MADDPG_same', 'NL_DDPG_same', 'NL_MADDPG_same', 'NG_DDPG_same', 'NG_MADDPG_same'],
             ['CG_DDPG_woCur', 'CG_MADDPG_woCur', 'CL_DDPG_woCur', 'CL_MADDPG_woCur', 'NG_DDPG_woCur', 'NG_MADDPG_woCur', 'NL_DDPG_woCur', 'NL_MADDPG_woCur'],
             ['CG_DDPG_pow1', 'CG_MADDPG_pow1', 'CG_DDPG_same_pow1', 'CG_MADDPG_same_pow1', 'CL_DDPG_pow1', 'CL_MADDPG_pow1', 'CL_DDPG_same_pow1', 'CL_MADDPG_same_pow1'],
             ['CL_DDPG_org_5PMD', 'CL_MADDPG_org_5PMD', 'CG_DDPG_org_5PMD', 'CG_MADDPG_org_5PMD', 'NL_DDPG_5PMD', 'NL_MADDPG_5PMD', 'NG_DDPG_5PMD', 'NG_MADDPG_5PMD',
              'CL_DDPG_org_woCur_5PMD', 'CL_MADDPG_org_woCur_5PMD', 'CG_DDPG_org_woCur_5PMD', 'CG_MADDPG_org_woCur_5PMD', 'NL_DDPG_woCur_5PMD', 'NL_MADDPG_woCur_5PMD', 'NG_DDPG_woCur_5PMD', 'NG_MADDPG_woCur_5PMD','CL_DDPG_org_5PMD_pow0']]
TestCases = [[],[],[],[],[],['CG_DDPG_org_5PMD_half_force','CG_DDPG_org_5PMD', 'CG_DDPG_commnoforce', 'CG_DDPG_comWithInvalidData_wforce', 'CG_DDPG_org_woCur_5PMD', 'NG_MADDPG_woCur_5PMD', 'NG_MADDPG_5PMD_woCur_noEval', 'NG_DDPG_woCur_5PMD']]
max_mean_r = [280, 500, 500, 500, 600, 600]
max_stacked_c = [1000, 2000, 2000, 2000, 2000, 2000]
max_simultaneous_c = [4.0, 20.0, 4.0, 20.0, 20.0, 20.0]
offset = [1.3, 8, 8, 8, 8, 8]

TCS = TestCases[test_type]
MMR = max_mean_r[test_type]
MSC = max_stacked_c[test_type]
MSSC = max_simultaneous_c[test_type]
OFS = offset[test_type]

res_dir = './results/'
MIN_EP = 0
MAX_EP = 100000
Required_benchmark_counts = 10
Required_traind_episodes = 100 #save_rate = 1000回に1回ベンチマークだとして

#TCS=['NG_MADDPG_5PMD_woCur_noEval']

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv2.vconcat(im_list_resize)

for TC in TCS: #各ベンチマークディレクトリ毎に
    if os.path.isdir(res_dir+TC): #ディレクトリがあることを確認、なければスキップ
        files = os.listdir(res_dir+TC) #ファイルリスト
        targetfiles = [l for l in files if ('benchmark' in l and '.csv' in l and 'lock' not in l)] #benchmarkと.csvを含むファイルのみを抽出
        df = pd.DataFrame([])

        for count, f in enumerate(targetfiles):
            #print ('  processing file:',f)
            fname = res_dir+ TC + '/' + f
            dft = pd.read_csv(fname, index_col=False, header=0, engine='python', dtype='str')
            var_lst = ['Global_counter','Episodes','Follower1','Follower2','Follower3','Leader','Prey','Mutual Collision']
            dft[var_lst] = dft[var_lst].astype(float)
            dft[var_lst] = dft[var_lst].astype(int)
            var_lst = ['mean rew F1',' mean rew F2',' mean rew F3',' mean rew L',' mean rew Prey',' mean rew total',' time']
            dft[var_lst] = dft[var_lst].astype(float)
            df = pd.concat([df, dft])
            
            #adding best scores of each trials
            dft['adv_sum'] = dft['Follower1'] + dft['Follower2'] + dft['Follower3'] + dft['Leader']
            maxarg=dft['adv_sum'].idxmax()
            print (fname, 'adv:',dft['adv_sum'].max(), 'at episode', dft.loc[maxarg, 'Episodes'], 'simul:', dft['Mutual Collision'].max(), 'at episode', dft.loc[dft['Mutual Collision'].idxmax(), 'Episodes'])
            
        df.reset_index(inplace=True,drop=True)
        max_adv = df['Prey'].max()
        max_adv_at = df['Prey'].idxmax()
        max_simul = df['Mutual Collision'].max()
        max_simul_at = df['Mutual Collision'].idxmax()
        gcadv = df.loc[max_adv_at,'Global_counter']
        gcsimul = df['Global_counter'].iloc[max_simul_at]
        print ('@@@@@@@@@ BEST at GC', gcadv, 'adv', max_adv, 'at eps', df.loc[max_adv_at, 'Episodes'])
        print ('@@@@@@@@@ BEST at GC', gcsimul, 'simul', max_simul, 'at eps', df.loc[max_simul_at, 'Episodes'])

                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
