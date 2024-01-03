import os
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import copy

test_type = 4 # 0: normal, 1: same, 2: without_Curriculum, 3: pow1, 4: original implementation and prey_maintain_duration=5

TestCases = [['CL_DDPG', 'CL_MADDPG', 'CG_DDPG', 'CG_MADDPG', 'NL_DDPG', 'NL_MADDPG', 'NG_DDPG', 'NG_MADDPG'],
             ['CL_DDPG_same', 'CL_MADDPG_same', 'CG_DDPG_same', 'CG_MADDPG_same', 'NL_DDPG_same', 'NL_MADDPG_same', 'NG_DDPG_same', 'NG_MADDPG_same'],
             ['CG_DDPG_woCur', 'CG_MADDPG_woCur', 'CL_DDPG_woCur', 'CL_MADDPG_woCur', 'NG_DDPG_woCur', 'NG_MADDPG_woCur', 'NL_DDPG_woCur', 'NL_MADDPG_woCur'],
             ['CG_DDPG_pow1', 'CG_MADDPG_pow1', 'CG_DDPG_same_pow1', 'CG_MADDPG_same_pow1', 'CL_DDPG_pow1', 'CL_MADDPG_pow1', 'CL_DDPG_same_pow1', 'CL_MADDPG_same_pow1'],
             ['CL_DDPG_org_5PMD', 'CL_MADDPG_org_5PMD', 'CG_DDPG_org_5PMD', 'CG_MADDPG_org_5PMD', 'NL_DDPG_5PMD', 'NL_MADDPG_5PMD', 'NG_DDPG_5PMD', 'NG_MADDPG_5PMD',
              'CL_DDPG_org_woCur_5PMD', 'CL_MADDPG_org_woCur_5PMD', 'CG_DDPG_org_woCur_5PMD', 'CG_MADDPG_org_woCur_5PMD', 'NL_DDPG_woCur_5PMD', 'NL_MADDPG_woCur_5PMD', 'NG_DDPG_woCur_5PMD', 'NG_MADDPG_woCur_5PMD']]
max_mean_r = [280, 500, 500, 500, 600]
max_stacked_c = [1000, 2000, 2000, 2000, 2000]
max_simultaneous_c = [4.0, 20.0, 4.0, 20.0, 20.0]
offset = [1.3, 8, 8, 8, 8]

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


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv2.vconcat(im_list_resize)

for TC in TCS: #各ベンチマークディレクトリ毎に
    if os.path.isdir(res_dir+TC): #ディレクトリがあることを確認、なければスキップ
        files = os.listdir(res_dir+TC) #ファイルリスト
        targetfiles = [l for l in files if ('benchmark' in l and '.csv' in l and 'lock' not in l)] #benchmarkと.csvを含むファイルのみを抽出
        df = pd.DataFrame([])
        print ('processing', TC)
        for f in targetfiles:
            #print ('  processing file:',f)
            fname = res_dir+ TC + '/' + f
            dft = pd.read_csv(fname, index_col=False, header=0, engine='python', dtype='str')
            var_lst = ['Global_counter','Episodes','Follower1','Follower2','Follower3','Leader','Prey','Mutual Collision']
            dft[var_lst] = dft[var_lst].astype(float)
            dft[var_lst] = dft[var_lst].astype(int)
            var_lst = ['mean rew F1',' mean rew F2',' mean rew F3',' mean rew L',' mean rew Prey',' mean rew total',' time']
            dft[var_lst] = dft[var_lst].astype(float)
            df = pd.concat([df, dft])
        
        gc = df['Global_counter'].unique() #結合したベンチマークデータからユニークなglobal_counterを抽出
        number_of_data = len(gc)
        gc = np.sort(gc) #ソート
        if (number_of_data != Required_benchmark_counts):
            print('#### ERROR! ####   ',TC,'has only',number_of_data,'of trained benchmark data. There are only', gc)
        for g in gc:
            actual_data_length = len(df[df['Global_counter'] ==g])
            if (actual_data_length != Required_traind_episodes):
                print ('#### ERROR! ####  global counter:',g,'has only length of',actual_data_length)
        
        #Deepmind風グラフ-5000以上
        #fig, ax1 = plt.subplots()
        fig, (ax1, ax3, ax4) = plt.subplots(nrows=3, figsize=(8,8), sharex=True, gridspec_kw={'height_ratios':[3,3,1]})
        ax1.grid(True)
        #ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean rewards of Predators')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mean reward of Prey')
        ax1.set_xlim(MIN_EP,MAX_EP)
        ax2.set_xlim(MIN_EP,MAX_EP)
        ax1.set_ylim(0, MMR)
        if False:#'G_' in TC: #global reward -> rewards of predators are same -> now always different, not use here
            rew_names = [' mean rew L',' mean rew Prey']
            labels = ['Predator','Prey']
            colors=np.array(['blue', 'green'])
        else: #local reward
            rew_names = ['mean rew F1',' mean rew F2',' mean rew F3',' mean rew L','Mutual Collision',' mean rew Prey']
            labels = ['Follower1', 'Follower2', 'Follower3', 'Leader','Simultaneous Collisions','Prey']
            colors=np.array(['grey', 'red', 'purple', 'blue', 'green'])
            
        # plot of reward
        all_means = pd.DataFrame([]) 
        all_std = pd.DataFrame([]) 
        for i in range(0, MAX_EP, 1000):
            tdf = df[df['Episodes'] ==i]
            temp = pd.DataFrame(np.array([tdf[rew_names[l]].mean() for l in range(len(rew_names))]).reshape(1,len(rew_names)), columns=[labels[k] for k in range(len(rew_names))], index=[i])
            all_means = all_means.append(temp)
            temp = pd.DataFrame(np.array([tdf[rew_names[l]].std(ddof=0) for l in range(len(rew_names))]).reshape(1,len(rew_names)), columns=[labels[k] for k in range(len(rew_names))], index=[i])
            all_std = all_std.append(temp)
        all_means = all_means[all_means.index>MIN_EP]
        all_std =all_std[all_std.index>MIN_EP]
        for name, clr, lbl in zip(labels[:-1], colors[:-1], labels[:-1]):
            ax1.plot(all_means.index, all_means[name], c=clr, alpha=1.0, label=lbl)
            ax1.fill_between(all_std.index, all_means[name]-all_std[name], all_means[name]+all_std[name], color=clr, alpha=0.2)
        ax2.plot(all_means.index, all_means[labels[-1]], c=colors[-1], alpha = 0.5, label=labels[-1])
        ax2.fill_between(all_std.index, all_means[labels[-1]]-all_std[labels[-1]], all_means[labels[-1]]+all_std[labels[-1]], color=colors[-1], alpha = 0.1)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='best')
        ax1.set_title(TC+'\n Mean rewards')
        #plt.savefig(TC+'_rewards.png', figsize=(24, 18), dpi=300, bbox_inches="tight")
        
        #Deepmind風グラフ-5000以上,　衝突回数-------------------------------------------
        #fig, ax1 = plt.subplots()
        ax3.grid(True)
        ax3.set_ylabel('Mean collision counts')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        names = ['Follower1','Follower2','Follower3','Leader','Mutual Collision']
        labels =['Follower1','Follower2','Follower3','Leader','Simultaneous Collisions']
        alphas = [0.5, 0.5, 0.5, 0.5, 1.0]
        colors=np.array(['grey', 'red', 'purple', 'blue', 'black'])
        
        # plot of all means
        all_means = pd.DataFrame([]) 
        for i in range(0, MAX_EP, 1000):
            tdf = df[df['Episodes'] ==i]
            temp = pd.DataFrame(np.array([tdf[names[l]].mean() for l in range(len(names))]).reshape(1,len(names)), columns=[labels[k] for k in range(len(names))], index=[i])
            all_means = all_means.append(temp)
        all_means = all_means[all_means.index>MIN_EP]
        
        stacked = all_means[labels[0]]
        past = copy.deepcopy(stacked)
        ax3.plot(all_means.index, stacked, c=colors[0], alpha=alphas[0], label=labels[0])
        ax3.fill_between(all_means.index, [0 for l in range(len(all_means.index))], stacked, color=colors[0], alpha=0.2)
        for name, clr, lbl, al in zip(labels[1:-1], colors[1:-1], labels[1:-1], alphas[1:-1]):
            stacked += all_means[lbl]
            ax3.plot(all_means.index, stacked, c=clr, alpha=al, label=lbl)
            ax3.fill_between(all_means.index, past, stacked, color=clr, alpha=0.2)
            past = copy.deepcopy(stacked)
        '''
        for name, clr, lbl, al in zip(labels[:-1], colors[:-1], labels[:-1], alphas[:-1]):
            ax3.plot(all_means.index, all_means[name], c=clr, alpha=al, label=lbl)
        ax3.fill_between(all_means.index, [0 for l in range(len(all_means.index))], all_means[labels[4]], color='grey', alpha=0.1)
        '''
        
        #最大値のところにマーク
        maxx = stacked.idxmax()
        maxy = stacked.max()
        arrow_dict = dict(arrowstyle = "->", color="black", connectionstyle = "angle, angleA = 0, angleB = 90")
        text_dict = dict(boxstyle = "round", fc='white', ec='black')
        ax3.annotate("Max:"+'{:.0f}'.format(maxy), xy = (maxx, maxy), xytext=(maxx-7000, maxy+200), bbox=text_dict, arrowprops=arrow_dict)
        ax3.set_ylim(0, MSC) #maxy+200)
        ax3.legend(loc='best')
        ax3.set_title('Stacked mean collision counts of each Predators')

        #Nomber 3 graph - simultaneous collisions
        ax4.set_title('Mean simultaneous collision counts')
        ax4.set_ylabel('counts')
        maxx = all_means[labels[-1]].idxmax()
        maxy = all_means[labels[-1]].max() #+ all_std[labels[-1]].loc[maxx]
        if (maxy == 0): #There is no simultaneous counts
            maxy = 1.0
        else:
           ax4.annotate("Max:"+'{:.2f}'.format(maxy), xy = (maxx, maxy), xytext=(maxx-7000, maxy+OFS), bbox=text_dict, arrowprops=arrow_dict)
        ax4.set_ylim(0, MSSC) #maxy*1.2)
        ax4.plot(all_means.index, all_means[labels[-1]], c=colors[-1], alpha = 0.5, label=labels[-1])
        ax4.fill_between(all_std.index, all_means[labels[-1]]-all_std[labels[-1]], all_means[labels[-1]]+all_std[labels[-1]], color=colors[-1], alpha = 0.1)
        ax4.legend(loc='upper left')
        ax4.set_xlabel('Episodes')
        
        fig.text(0.96, 0.02, number_of_data)

        #plt.savefig(TC+'_collision_counts.png', figsize=(24, 18), dpi=300, bbox_inches="tight")
        plt.savefig('./results/'+TC+'.png', figsize=(24, 18), dpi=300)
        plt.close()
        '''
        im1 = cv2.imread(TC+'_rewards.png')
        im2 = cv2.imread(TC+'_collision_counts.png')
        im_v = vconcat_resize_min([im1, im2])
        cv2.imwrite(TC+'.png', im_v)
        os.remove(TC+'_rewards.png')
        os.remove(TC+'_collision_counts.png')
        '''
        

