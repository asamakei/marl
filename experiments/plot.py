import csv
import math
import numpy as np
import matplotlib.pyplot as plt
'''
csv_path = 'results/OneSuperLeader_MultiTarget1/OneSuperLeader_MultiTarget1_benchmark_G0.csv'
rows = []
with open(csv_path) as f:
    reader = csv.reader(f)
    rows = [[row[1],row[10],row[11],row[16]] for row in reader]
header = rows.pop(0)
episode = np.array([row[0] for row in rows],dtype='int32')
score = np.array([row[1] for row in rows],dtype='float')+np.array([row[2] for row in rows],dtype='float')
stdev = np.array([math.sqrt(float(row[3]))/2 for row in rows],dtype='float')
score_low = score-stdev
score_upp = score+stdev
score = score/2
score_low = score_low/2
score_upp = score_upp/2
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel("number of episode", fontsize = 14)
ax.set_ylabel("mean reward of Followers", fontsize = 14)
ax.set_xlim(0,episode[-1])
ax.set_ylim(np.min(score_low[5:])-20,np.max(score_upp[5:])+20)

#ax.plot(episode,score_low,color="#0000FF",alpha=0.2)
#ax.plot(episode,score_upp,color="#0000FF",alpha=0.2)
#ax.fill_between(episode,score_low,score_upp,facecolor='#0000FF', alpha=0.1)
ax.plot(episode,score,color="#0000FF")


csv_path = 'results/OneLeader_MultiTarget1/OneLeader_MultiTarget1_benchmark_G0.csv'
rows = []
with open(csv_path) as f:
    reader = csv.reader(f)
    rows = [[row[1],row[10],row[11],row[12],row[13],row[16]] for row in reader]
header = rows.pop(0)
episode = np.array([row[0] for row in rows],dtype='int32')
score = np.array([row[1] for row in rows],dtype='float')+np.array([row[2] for row in rows],dtype='float')+np.array([row[3] for row in rows],dtype='float')+np.array([row[4] for row in rows],dtype='float')
stdev = np.array([math.sqrt(float(row[5]))/2 for row in rows],dtype='float')
score_low = score-stdev
score_upp = score+stdev
score = score/4
score_low = score_low/4
score_upp = score_upp/4
#ax.plot(episode,score_low,color="#FF0000",alpha=0.2)
#ax.plot(episode,score_upp,color="#FF0000",alpha=0.2)
#ax.fill_between(episode,score_low,score_upp,facecolor='#FF0000', alpha=0.1)
ax.plot(episode,score,color="#FF0000")

plt.savefig("score.png")
plt.show()
'''
def make_graph(scenario_name,exp_num,data_rows,div,color):
    csv_path = 'results/'+scenario_name+str(exp_num)+"/"+scenario_name+str(exp_num)+"_benchmark_G0.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = [[row[i] for i in data_rows] for row in reader]
    header = rows.pop(0)
    episode = np.array([row[0] for row in rows],dtype='int32')
    scores = []
    for i in range(len(data_rows)-2):
        scores.append(np.array([row[i+1] for row in rows],dtype='float'))
    
    score = scores[0]
    for i in range(len(scores)-1):
        score = score + scores[i+1]
    
    stdev = np.array([math.sqrt(float(row[-1]))/div for row in rows],dtype='float')
    score_low = score-stdev
    score_upp = score+stdev
    score = score
    score_low = score_low
    score_upp = score_upp


    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel("number of episode", fontsize = 14)
    ax.set_ylabel("reward of Followers", fontsize = 14)
    ax.set_xlim(0,episode[-1])
    ax.set_ylim(np.min(score_low[3:])-20,np.max(score_upp[3:])+20)
    ax.set_ylim(np.min(score_low[3:])-20,71)
    ax.plot(episode,score_low,color=color,alpha=0.2)
    ax.plot(episode,score_upp,color=color,alpha=0.2)
    ax.fill_between(episode,score_low,score_upp,facecolor=color, alpha=0.1)
    ax.plot(episode,score,color=color)    

    plt.savefig('results/'+scenario_name+str(exp_num)+"/"+scenario_name+str(exp_num)+".png")

def make_two_graphs(scenario_name1,scenario_name2):
    csv_path = 'results/'+scenario_name1+str(0)+"/"+scenario_name1+str(0)+"_benchmark_G0.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = [[row[i] for i in [1,8,9,12]] for row in reader]
    header = rows.pop(0)
    episode = np.array([row[0] for row in rows],dtype='int32')
    scores = []
    for i in range(2):
        scores.append(np.array([row[i+1] for row in rows],dtype='float'))
    
    score = scores[0]
    for i in range(len(scores)-1):
        score = score + scores[i+1]
    
    stdev = np.array([math.sqrt(float(row[-1]))/2 for row in rows],dtype='float')
    score_low = score-stdev
    score_upp = score+stdev
    score = score
    score_low = score_low
    score_upp = score_upp

    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel("number of episode", fontsize = 14)
    ax.set_ylabel("reward of Followers", fontsize = 14)
    ax.set_xlim(0,episode[-1])
    ax.set_ylim(np.min(score_low[3:])-20,np.max(score_upp[3:])+20)

    ax.plot(episode,score_low,color='red',alpha=0.2)
    ax.plot(episode,score_upp,color='red',alpha=0.2)
    ax.fill_between(episode,score_low,score_upp,facecolor='red', alpha=0.1)
    ax.plot(episode,score,color='red')

    csv_path = 'results/'+scenario_name2+str(0)+"/"+scenario_name2+str(0)+"_benchmark_G0.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = [[row[i] for i in [1,10,11,16]] for row in reader]
    header = rows.pop(0)
    episode = np.array([row[0] for row in rows],dtype='int32')
    scores = []
    for i in range(2):
        scores.append(np.array([row[i+1] for row in rows],dtype='float'))
    
    score = scores[0]
    for i in range(len(scores)-1):
        score = score + scores[i+1]
    
    stdev = np.array([math.sqrt(float(row[-1]))/3 for row in rows],dtype='float')
    score_low = score-stdev
    score_upp = score+stdev
    score = score
    score_low = score_low
    score_upp = score_upp

    ax.plot(episode,score_low,color='blue',alpha=0.2)
    ax.plot(episode,score_upp,color='blue',alpha=0.2)
    ax.fill_between(episode,score_low,score_upp,facecolor='blue', alpha=0.1)
    ax.plot(episode,score,color='blue')

    plt.savefig('results/test.png')

#make_graph("Linear3",1,[1,8,12],4,'red')
#make_graph("ManyTargetL1",0,[1,8,9,12],2,'red')
#make_graph("ManyTargetS1",0,[1,10,11,16],3,'red')
#make_graph("Linear2",1,[1,7,10],2,'red')
make_two_graphs("ManyTargetL1","ManyTargetS1")
#make_graph("OneSuperLeader_MultiTarget105",0,[1,10,11,16],3,'red')
#make_two_graphs("OneSuperLeader_MultiTarget105","L1F2_MultiTarget105_")
#make_graph("L1F2_MultiTarget090_",0,[1,8,9,12],2,'red')
#make_graph("L1F2_MultiTarget110_",0,[1,8,9,12],2,'red')
#make_graph("L1F2_MultiTarget075_",0,[1,8,9,12],2,'red')
#make_graph("L1F2_MultiTarget125_",0,[1,8,9,12],2,'red')
#make_graph("L1F2_MultiTarget095_",0,[1,8,9,12],2,'red')
#make_graph("L1F2_MultiTarget105_",0,[1,8,9,12],2,'red')