import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

YMIN, YMAX = 0.35, 1.0
YMIN_1 = 0.55

DOES_INCLUDE_NONSTART_S0 = True

ys_dict = {}
es_dict = {}

HARDNESS_LIST = ['Easy', 'Hard']
MAP_ABBR_DICT = {'Easy': 'e', 'Med':'m', 'Hard':'h'}

avg_ys_hardness_dict = {}
avg_es_hardness_dict = {}
avg_s0 = 0.

avg_ys_hardness_dict_1 = {}
avg_es_hardness_dict_1 = {}

MAP_NUM_TEST_TRIALS = {'e': '9','m': '7', 'h': '8'}

NUM_EPISODES = 49
MAX_NUM_FALCON_INPUT = 10
s0_NAME = 'NoneStartModel Trained with data from Sparky Map 0'

avg_arima = np.zeros(MAX_NUM_FALCON_INPUT)
for MAP_HARDNESS in HARDNESS_LIST:
    MAP_ABBR = MAP_ABBR_DICT[MAP_HARDNESS]
    arima_PATH_1 = '../ARIMA-master/results/f' +  MAP_ABBR + '/train'
    arima_PATH_2 = '/mae.npy'
    for i in range(1, MAX_NUM_FALCON_INPUT+1):
        avg_arima[i-1] += 1.0 - np.load(arima_PATH_1 + str(i) + arima_PATH_2)
avg_arima /= len(HARDNESS_LIST)

rule_base = {'e': 0.8175, 'm': 0.8113, 'h': 0.8074}
avg_rule_base = np.zeros(MAX_NUM_FALCON_INPUT)
for MAP_HARDNESS in HARDNESS_LIST:
    MAP_ABBR = MAP_ABBR_DICT[MAP_HARDNESS]
    for i in range(MAX_NUM_FALCON_INPUT):
        avg_rule_base[i] += rule_base[MAP_ABBR]
avg_rule_base /= len(HARDNESS_LIST)


for MAP_HARDNESS in HARDNESS_LIST:
    MAP_ABBR = MAP_ABBR_DICT[MAP_HARDNESS]
    NUM_TEST = MAP_NUM_TEST_TRIALS[MAP_ABBR]
    s0_PATH = 'data/results/room_level/f' + MAP_ABBR + 'fm/batchsize4/trainval-fm_test-f' + MAP_ABBR + MAP_NUM_TEST_TRIALS[MAP_ABBR] + 'trials/test_loss_all.npy'

    ys_hardness_dict = {}
    es_hardness_dict = {}

    ys_hardness_dict_1 = {}
    es_hardness_dict_1 = {}

    if DOES_INCLUDE_NONSTART_S0:
        # names.append(s0_NAME)
        # data.append(np.load(s0_PATH))
        tmp = np.load(s0_PATH)
        avg_s0 += (np.mean(tmp[:, 0, 0:4])/len(HARDNESS_LIST))


    for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:

        f_NAME = 'StartModel s0 '
        if not DOES_TRAIN_FROM_SAVED_MODEL:
            f_NAME = 'NoneStartModel '
        f_NAME += 'Trained with testbed Falcon '

        f_PATH_1 = 'data/results/room_level/f' + MAP_ABBR + 'fm/batchsize4/'
        # f_PATH_1 = 'data/results/grid_level/s0f' + MAP_ABBR + '/batchsize4/'
        if not DOES_TRAIN_FROM_SAVED_MODEL:
            f_PATH_1 += 'non'
        f_PATH_1 += 'savedmodel-fm_trainval-f' + MAP_ABBR
        f_PATH_2 = 'trials_test-f' + MAP_ABBR + MAP_NUM_TEST_TRIALS[MAP_ABBR] + 'trials/test_loss_all.npy'



        # set up names and load data according to names
        names = []
        x = np.array(range(NUM_EPISODES))
        data = []

        for i in range(1, MAX_NUM_FALCON_INPUT+1):
            names.append(f_NAME  + MAP_HARDNESS +  ' ' + str(i) + ' training trajectories')
            data.append(np.load(f_PATH_1 + str(i) + f_PATH_2))

        # prepare for the plot
        ys = [[] for _ in range(len(names))]
        es = [[] for _ in range(len(names))]
        for t in range(NUM_EPISODES):
            # the data is the loss = mae, and only the first 4 refers to falcon map.
            # this gives the prediction accuracy
            for i in range(len(names)):
                ys[i].append(1 - np.mean(data[i][t, 0, 0:4]))
                # ys[i].append(1 - np.mean(data[i][t, 0, 4:8]))
                es[i].append(np.std(data[i][t, 0, :]))

        # # make the plot
        # fig = plt.figure(figsize=(7, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # for i in range(len(names)):
        #     plt.errorbar(x, ys[i], es[i], linestyle='-', marker='.', label=names[i])
        # plt.legend(title='Prediction Accuracy (1.0 - MAE)')
        # plt.xlabel("Episodes")
        # plt.ylabel("Accuracy")
        # ax.set_ylim(-0.1, 1.0)
        # plt.title('Test on Falcon Map ' + MAP_HARDNESS + ' Test trajectories')
        # plt.grid(True)
        # plt.show()

        # print out the final accuracies
        ys_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL] = []
        es_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL] = []
        ys_hardness_dict_1[DOES_TRAIN_FROM_SAVED_MODEL] = []
        es_hardness_dict_1[DOES_TRAIN_FROM_SAVED_MODEL] = []
        for i in range(len(names)):
            # print (names[i], ys[i][-1])
            ys_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(ys[i][-20:-1]))
            es_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(es[i][-20:-1]))
            # ys_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(ys[i]))
            # es_hardness_dict[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(es[i]))
        for t in range(NUM_EPISODES):
            ys_array = np.array(ys)
            es_array = np.array(es)
            ys_hardness_dict_1[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(ys_array[:,t]))
            es_hardness_dict_1[DOES_TRAIN_FROM_SAVED_MODEL].append(np.mean(es_array[:,t]))
        # ys_dict[MAP_HARDNESS] = ys[0][-1]
        # es_dict[MAP_HARDNESS] = es[0][-1]

    avg_ys_hardness_dict[MAP_HARDNESS] = ys_hardness_dict
    avg_es_hardness_dict[MAP_HARDNESS] = es_hardness_dict

    avg_ys_hardness_dict_1[MAP_HARDNESS] = ys_hardness_dict_1
    avg_es_hardness_dict_1[MAP_HARDNESS] = es_hardness_dict_1


    # # fig = plt.figure(figsize=(7, 4))
    # ind = np.arange(MAX_NUM_FALCON_INPUT)
    # # width = 0.35
    # # plt.bar(ind, ys_hardness_dict[False], width, label='Non-startmodel ')
    # # plt.bar(ind + width, ys_hardness_dict[True], width, label='startmodel is s0')
    # fig, ax = plt.subplots(figsize=(7, 4))
    #
    # print (list(ind) + list(ind))
    # print (['F']*len(ind) + ['M']*len(ind))
    # print (ys_hardness_dict[False] + ys_hardness_dict[True])
    # df = pd.DataFrame({'Number of trajectories for finetuning': list(ind) + list(ind), 'Type': ['With finetuning (use pre-trained model with Sparky Data)']*len(ind) + ['Trained from Scratch']*len(ind), 'Accuracy': ys_hardness_dict[True] + ys_hardness_dict[False]})
    # palette = ['b', 'r']
    # sns.lineplot(x="Number of trajectories for finetuning", y="Accuracy",
    #              hue="Type", style="Type",markers=True, palette=palette,
    #              data=df)
    # for item, color in zip(df.groupby('Type'), palette):
    #     #item[1] is a grouped data frame
    #     for x,y,m in item[1][['Number of trajectories for finetuning','Accuracy','Type']].values:
    #         ax.text(x,y,f'{y:.4f}',color='black')
    #
    #
    # # # print (data)
    # plt.grid(True)
    # plt.show()
    # #
    # # plt.ylabel('Final Accuracies')
    # # plt.xlabel('Number of Input Falcon Trajectories')
    # # plt.title('Final Accuracies Varying Number of Trajectory Inputs Test on Falcon Map ' + MAP_HARDNESS)
    # #
    # # plt.xticks(ind + width / 2, [str(i+1) for i in ind])
    # # plt.legend(loc='lower right')
    # # plt.grid(True)
    # # plt.show()

# average over different maps
avg_ys = {}
avg_es = {}
# print ('avgs0', avg_s0)
for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:
    avg_ys[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(MAX_NUM_FALCON_INPUT)
    avg_es[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(MAX_NUM_FALCON_INPUT)
    # print ('avg_ys[DOES_TRAIN_FROM_SAVED_MODEL]', avg_ys[DOES_TRAIN_FROM_SAVED_MODEL])
    for MAP_HARDNESS in HARDNESS_LIST:
        avg_ys[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_ys_hardness_dict[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)
        avg_es[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_es_hardness_dict[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)
        # print ('avg_ys[DOES_TRAIN_FROM_SAVED_MODEL]', avg_ys[DOES_TRAIN_FROM_SAVED_MODEL])
        # print ()

# average over different maps
avg_ys_1 = {}
avg_es_1 = {}
for DOES_TRAIN_FROM_SAVED_MODEL in [False, True]:
    avg_ys_1[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(NUM_EPISODES)
    avg_es_1[DOES_TRAIN_FROM_SAVED_MODEL] = np.zeros(NUM_EPISODES)
    for MAP_HARDNESS in HARDNESS_LIST:
        avg_ys_1[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_ys_hardness_dict_1[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)
        avg_es_1[DOES_TRAIN_FROM_SAVED_MODEL] += np.array(avg_es_hardness_dict_1[MAP_HARDNESS][DOES_TRAIN_FROM_SAVED_MODEL])/len(HARDNESS_LIST)
        # print ('avg_ys[DOES_TRAIN_FROM_SAVED_MODEL]', avg_ys[DOES_TRAIN_FROM_SAVED_MODEL])
        # print ()



ind = np.arange(MAX_NUM_FALCON_INPUT)
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_ylim([YMIN, YMAX])

# y_name = 'Average Test Accuracy (over 50 Episodes)'
y_name = 'Test Accuracy after Convergence'

# print (list(avg_ys[True]) + list(avg_ys[False]))
df = pd.DataFrame({'Number of Finetuning Trajectories': [0] + list(ind+1) + list(ind+1) + list(ind+1) + list(ind+1), \
                    'Type': ['With finetuning (use pre-trained model with Falcon Med Data)']* (1 +len(ind)) + ['Trained from Scratch']*len(ind) + ['ARIMA']*len(ind) + ['Rule-based']*len(ind),\
                    y_name: [avg_s0] + list(avg_ys[True]) + list(avg_ys[False]) + list(avg_arima) + list(avg_rule_base),\
                    'lable y': [avg_s0] + list(avg_ys[True]+0.01) + list(avg_ys[False]-0.03) + list(avg_arima-0.025) + list(avg_rule_base-0.03)})
palette = ['b', 'r', 'g', 'brown']
sns.lineplot(x="Number of Finetuning Trajectories", y=y_name,
             hue="Type", style="Type",markers=True, palette=palette,
             data=df)
for item, color in zip(df.groupby('Type'), palette):
    for x,y,y_label,m in item[1][['Number of Finetuning Trajectories',y_name,'lable y','Type']].values:
        # ax.text(x,y_label,f'{y:.3f}',color='black')
        pass

plt.grid(True)
plt.title('Test Accuracy wrt Number of Finetuning Trajectories \n(Graph Level, Test on Falcon Easy and Hard Maps)')
plt.show()

# x_pos = [i for i, _ in enumerate(HARDNESS_LIST)]
# fig = plt.figure(figsize=(6, 4))
# y = [ys_dict[hardness] for hardness in HARDNESS_LIST]
# e = [es_dict[hardness] for hardness in HARDNESS_LIST]
# plt.bar(x_pos, y, color='yellow', yerr=e)
# plt.xlabel("Map Hardness")
# plt.ylabel("Accuracies")
# plt.title("Final Accuracies Testing on Falcon Map Test trajectories with Model Trained with Sparky Trajectories")
# plt.xticks(x_pos, [HARDNESS_LIST[i] + '\n' + str(round(y[i], 4)) for i in range(len(y))])
# plt.show()

PLOT_NUM_EPI = NUM_EPISODES-10
ind_1 = np.arange(PLOT_NUM_EPI)
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_ylim([YMIN_1, YMAX])

# print (list(avg_ys[True]) + list(avg_ys[False]))
df_1 = pd.DataFrame({'Training Episodes': list(ind_1+1) + list(ind_1+1) , \
                    'Type': ['With finetuning (use pre-trained model with Falcon Med Data)']* len(ind_1) + ['Trained from Scratch']*len(ind_1) ,\
                    'Average Test Accuracy (among 1-10 Finetuning Trajectories)': list(avg_ys_1[True][0:PLOT_NUM_EPI]) + list(avg_ys_1[False][0:PLOT_NUM_EPI]) ,\
                    'lable y': list(avg_ys_1[True][0:PLOT_NUM_EPI]+0.001) + list(avg_ys_1[False][0:PLOT_NUM_EPI]-0.001) })
palette = ['b', 'r']
sns.lineplot(x="Training Episodes", y="Average Test Accuracy (among 1-10 Finetuning Trajectories)",
             hue="Type", style="Type",markers=True, palette=palette,
             data=df_1)
for item, color in zip(df_1.groupby('Type'), palette):
    for x,y,y_label,m in item[1][['Training Episodes','Average Test Accuracy (among 1-10 Finetuning Trajectories)','lable y','Type']].values:
        # ax.text(x,y_label,f'{y:.2f}',color='black')
        pass

plt.grid(True)
plt.title('Test Accuracy wrt Training Episodes \n(Graph Level, Test on Falcon Easy and Hard Maps)')
plt.show()
