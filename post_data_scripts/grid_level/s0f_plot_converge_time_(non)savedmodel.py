import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HARDNESS_LIST = ['Easy']
MAP_ABBR_DICT = {'Easy': 'e', 'Med':'m', 'Hard':'h'}

NUM_EXPERIMENTS = 10
CONVERGE_SUM = 0.1
CONS_COUNT = 20
MAP_NUM_TEST_TRIALS = {'e': '9','m': '7', 'h': '8'}

NUM_EPISODES = 49
MAX_NUM_FALCON_INPUT = 8

avg_ys_hardness_dict = {}
avg_ys_hardness_dict_1 = {}

for MAP_HARDNESS in HARDNESS_LIST:
    MAP_ABBR = MAP_ABBR_DICT[MAP_HARDNESS]
    NUM_TEST = MAP_NUM_TEST_TRIALS[MAP_ABBR]

# set up constants
# MAP_HARDNESS = 'Easy'
# MAP_ABBR = 'e'
    aas_hardness_dict = {}
    ec_hardness_dict = {}
    rt_hardness_dict = {}

    for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:
    # DOES_TRAIN_FROM_SAVED_MODEL = True
        f_NAME = 'StartModel s0 '
        if not DOES_TRAIN_FROM_SAVED_MODEL:
            f_NAME = 'NoneStartModel '
        f_NAME += 'Trained with testbed Falcon '

        f_PATH_1 = 'data/results/grid_level/s0f' + MAP_ABBR + '/batchsize4/'
        if not DOES_TRAIN_FROM_SAVED_MODEL:
            f_PATH_1 += 'non'
        f_PATH_1 += 'savedmodel-s0_trainval-f' + MAP_ABBR
        f_PATH_2 = 'trials_test-f' + MAP_ABBR + NUM_TEST + 'trials/'


        # set up names and load data according to names
        names = []
        x = np.array(range(NUM_EPISODES))
        data = []
        data_1 = []

        for i in range(1, MAX_NUM_FALCON_INPUT+1):
            names.append(f_NAME  + MAP_HARDNESS +  ' ' + str(i) + ' training trajectories')
            data.append(np.load(f_PATH_1 + str(i) + f_PATH_2 + 'test_loss_all.npy'))
            data_1.append(np.load(f_PATH_1 + str(i) + f_PATH_2 + 'runtime_all.npy'))

        ys = [[] for _ in range(len(names))]
        deltas = [[] for _ in range(len(names))]

        c = [0] * NUM_EXPERIMENTS
        for t in range(NUM_EPISODES):
            for i in range(len(names)):
                ys[i].append(1 - np.mean(data[i][t, 0, 0:4]))
            if t > 0 :
                if t > CONS_COUNT:
                    for i in range(len(names)):
                        if sum(abs(np.array(deltas[i][-CONS_COUNT:-1]))) > CONVERGE_SUM:
                            deltas[i].append(ys[i][t] - ys[i][t-1])
                else:
                    for i in range(len(names)):
                        deltas[i].append(ys[i][t] - ys[i][t-1])

        for i in range(len(names)):
            c[i] = len(deltas[i])

        # # prepare for the plot
        # fig = plt.figure(figsize=(7, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # plt.plot(c, linestyle='', marker='o')
        # plt.legend(title='Converge: sum(' + str(CONS_COUNT) + 'DeltaMAE) < ' + str(CONVERGE_SUM))
        # plt.xlabel("Number of Trajectory Input")
        # plt.ylabel("Number of Episodes Needed to Converge")
        #
        # ax.set_ylim(-0.1, 50.0)
        # plt.title('Test on Falcon Map ' + MAP_HARDNESS + ' Test trajectories')
        # plt.show()

        aas_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL] = []
        ec_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL] = []
        rt_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL] = []
        # print out the result
        print ('Average Accuracies Before Convergence')
        for i in range(len(names)):
            print (names[i], np.sum(ys[i][0: c[i]])/c[i])
            aas_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(np.sum(ys[i][0: c[i]])/c[i])

        print ('Episodes Needed towards Convergence')
        for i in range(len(names)):
            print (names[i], c[i])
            ec_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(c[i])

        for i in range(len(names)):
            runtime = 0
            for t in range(c[i]):
                runtime += data_1[i][t]
            rt_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(runtime)


    # fig = plt.figure(figsize=(6.5, 4))
    # ind = np.arange(MAX_NUM_FALCON_INPUT)
    # width = 0.35
    # plt.bar(ind, aas_hardness_dict[False], width, label='Non-startmodel ')
    # plt.bar(ind + width, aas_hardness_dict[True], width, label='startmodel is s0')
    #
    # plt.ylabel('Average Accuracies Before Convergence')
    # plt.xlabel('Number of Input Falcon Trajectories')
    # plt.title('Average Accuracies Before Convergence Varying Number of Trajectory Inputs \nTest on Falcon Map ' + MAP_HARDNESS)
    #
    # plt.xticks(ind + width / 2, [str(i+1) for i in ind])
    # plt.legend(loc='best')
    # plt.show()



    # fig = plt.figure(figsize=(6.5, 4))
    # ind = np.arange(MAX_NUM_FALCON_INPUT)
    # width = 0.35
    # plt.bar(ind, ec_hardness_dict[False], width, label='Non-startmodel ')
    # plt.bar(ind + width, ec_hardness_dict[True], width, label='startmodel is s0')
    #
    # plt.ylabel('Number of Episodes Taken for Convergence')
    # plt.xlabel('Number of Input Falcon Trajectories')
    # plt.title('Number of Episodes Taken for Convergence Varying Number of Trajectory Inputs \nTest on Falcon Map ' + MAP_HARDNESS)
    #
    # plt.xticks(ind + width / 2, [str(i+1) for i in ind])
    # plt.legend(loc='best')
    # plt.show()

    avg_ys_hardness_dict[MAP_HARDNESS] = ec_hardness_dict
    avg_ys_hardness_dict_1[MAP_HARDNESS] = rt_hardness_dict

# average over different maps
avg_ys = {}
# print ('avgs0', avg_s0)
for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:
    avg_ys[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(MAX_NUM_FALCON_INPUT)
    # print ('avg_ys[DOES_TRAIN_FROM_SAVED_MODEL]', avg_ys[DOES_TRAIN_FROM_SAVED_MODEL])
    for MAP_HARDNESS in HARDNESS_LIST:
        avg_ys[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_ys_hardness_dict[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)

# average over different maps
avg_ys_1 = {}
# print ('avgs0', avg_s0)
for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:
    avg_ys_1[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(MAX_NUM_FALCON_INPUT)
    # print ('avg_ys[DOES_TRAIN_FROM_SAVED_MODEL]', avg_ys[DOES_TRAIN_FROM_SAVED_MODEL])
    for MAP_HARDNESS in HARDNESS_LIST:
        avg_ys_1[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_ys_hardness_dict_1[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)






ind = np.arange(MAX_NUM_FALCON_INPUT)
fig, ax = plt.subplots(figsize=(7, 4))
# print (list(avg_ys[True]) + list(avg_ys[False]))
df = pd.DataFrame({'Number of Finetuning Trajectories': list(ind+1) + list(ind+1), \
                    'Type': ['With finetuning (use pre-trained model with Sparky Data)']* len(ind) + ['Trained from Scratch']*len(ind),\
                    'Average Number of Episodes Taken for Convergence': list(avg_ys[True]) + list(avg_ys[False]),\
                    'lable y': list(avg_ys[True]+0.01) + list(avg_ys[False]-0.01)})
palette = ['b', 'r']
sns.lineplot(x="Number of Finetuning Trajectories", y="Average Number of Episodes Taken for Convergence",
             hue="Type", style="Type",markers=True, palette=palette,
             data=df)
for item, color in zip(df.groupby('Type'), palette):
    for x,y,y_label,m in item[1][['Number of Finetuning Trajectories','Average Number of Episodes Taken for Convergence','lable y','Type']].values:
        ax.text(x,y_label,f'{y:.4f}',color='black')

plt.grid(True)
plt.title('Number of Episodes Taken for Convergence wrt Number of Trajectory Inputs \n(Grid Level, Test on Falcon Easy Map)')
plt.show()



ind = np.arange(MAX_NUM_FALCON_INPUT)
fig, ax = plt.subplots(figsize=(7, 4))
# print (list(avg_ys[True]) + list(avg_ys[False]))
df_1 = pd.DataFrame({'Number of Finetuning Trajectories': list(ind+1) + list(ind+1), \
                    'Type': ['With finetuning (use pre-trained model with Sparky Data)']* len(ind) + ['Trained from Scratch']*len(ind),\
                    'Average Runtime Taken for Convergence (seconds)': list(avg_ys_1[True]) + list(avg_ys_1[False]),\
                    'lable y': list(avg_ys_1[True]-200) + list(avg_ys_1[False]+200)})
palette = ['b', 'r']
sns.lineplot(x="Number of Finetuning Trajectories", y="Average Runtime Taken for Convergence (seconds)",
             hue="Type", style="Type",markers=True, palette=palette,
             data=df_1)
for item, color in zip(df_1.groupby('Type'), palette):
    for x,y,y_label,m in item[1][['Number of Finetuning Trajectories','Average Runtime Taken for Convergence (seconds)','lable y','Type']].values:
        ax.text(x,y_label,f'{y:.4f}',color='black')

plt.grid(True)
plt.title('Runtime Taken for Convergence wrt Number of Trajectory Inputs \n(Grid Level, Test on Falcon Easy Map)')
plt.show()
