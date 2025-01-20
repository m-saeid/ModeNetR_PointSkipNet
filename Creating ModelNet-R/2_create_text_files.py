import os

ds = '../data/modelnet40_normal_resampled'
r_clss = ['plant', 'flower_pot', 'vase', 'cup', 'bowl']

n_split_train = {'flower_pot': 189, 'plant': 114, 'vase': 599, 'cup': 34, 'bowl': 54} # ===
n_split_test = {'flower_pot': [190,262], 'plant': [115,153], 'vase': [600,722], 'cup': [35,43], 'bowl': [55,68]} # ===



            ################################
            ### Train : modelnet40_train ###
            ################################

train_split = open(f"{ds}/modelnet40_train.txt")
train_split = train_split.read()
train_split = train_split.split('\n')
#train_split.pop(0)

train_split2=[]
for name in train_split:
    if name[:-5] not in r_clss and name[:-5] != '':
        train_split2.append(name)

for clss in n_split_train:
    for i in range(1, n_split_train[clss]+1):
        train_split2.append(f'{clss}_{i:04d}')

train_split2.sort()

train_txt = ''
for i in train_split2:
    train_txt += f'{i}\n'

train_txt = train_txt[:-1]

with open(f'{ds}/modelnet40_train.txt', 'w') as file:
    file.write(train_txt)



            #############################
            ### Test: modelnet40_test ###
            #############################

test_split = open(f"{ds}/modelnet40_test.txt")
test_split = test_split.read()
test_split = test_split.split('\n')
#test_split.pop(0)

test_split2=[]
for name in test_split:
    if name[:-5] not in r_clss and name[:-5] != '':
        test_split2.append(name)

for clss in n_split_test:
    for i in range(n_split_test[clss][0], n_split_test[clss][1]+1):
        test_split2.append(f'{clss}_{i:04d}')

test_split2.sort()

test_txt = ''
for i in test_split2:
    test_txt += f'{i}\n'

test_txt = test_txt[:-1]

with open(f'{ds}/modelnet40_test.txt', 'w') as file:
    file.write(test_txt)


            ################
            ### filelist ###
            ################

all_data = train_split2 + test_split2
all_data.sort()

all_txt = ''
for i in all_data:
    all_txt += f'{i}\n'

all_txt = all_txt[:-1]

with open(f'{ds}/filelist.txt', 'w') as file:
    file.write(all_txt)

os.rename('../data/modelnet40_normal_resampled', '../data/modelnetR_normal_resampled')

print("Creating text files was done")