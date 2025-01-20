import pandas as pd
import random
import shutil
import torch
import os

random.seed = 0

print("""
      **************************************************************************************************
      If you need the information of the modelnet40_normal_resampled folder,
      be sure to make a backup copy.
      because the content of this folder will change and Modelnet40 will be converted to ModelnetR.
      **************************************************************************************************""")

file = 'replace_classes_5.xlsx'

flower_pot = pd.read_excel(file, 
                        sheet_name = 'flower_pot', 
                        index_col = 0)
 
plant = pd.read_excel(file, 
                        sheet_name = 'plant', 
                        index_col = 0)

vase = pd.read_excel(file, 
                        sheet_name = 'vase', 
                        index_col = 0)

cup = pd.read_excel(file, 
                        sheet_name = 'cup', 
                        index_col = 0)

bowl = pd.read_excel(file, 
                        sheet_name = 'bowl', 
                        index_col = 0)

ds = '../data/modelnet40_normal_resampled'

classes = {'flower_pot': flower_pot, 'plant':plant, 'vase': vase, 'cup': cup, 'bowl': bowl}

n_traint = {'flower_pot': 149, 'plant': 240, 'vase': 475, 'cup': 79, 'bowl': 64}
new_n_train = {'flower_pot': 149, 'plant': 240, 'vase': 475, 'cup': 79, 'bowl': 64}
#n_traint = {'flower_pot': [1,149], 'plant': [1,240], 'vase': [1,475], 'cup': [1,79], 'bowl': [1,64]} # <=
#n_test = {'flower_pot': [150,169], 'plant': [241,340], 'vase': [476,575], 'cup': [80,99], 'bowl': [65,84]}

r_clss = ['plant', 'flower_pot', 'vase', 'cup', 'bowl', 'remove']

data = dict.fromkeys(r_clss[:-1],[]) 

print()

# change the place of the point clouds based on the excel file
for src_cl in classes:
    for i in range(len(classes[src_cl])):
        for dst_cl in r_clss:
            if (classes[src_cl].iloc[i][dst_cl] == '*') and (src_cl != dst_cl):
                instns = f'{src_cl}_{i+1:04d}.txt'
                mode = 'train' if i+1<=n_traint[src_cl] else 'test'

                if dst_cl == 'remove':
                    os.remove(f'{ds}/{src_cl}/{instns}')
                    if mode == 'train':
                        new_n_train[src_cl] -= 1
                else:
                    #print(f'{src_cl}_{i+1}', dst_cl, mode)

                    shutil.move(f'{ds}/{src_cl}/{instns}', f'{ds}/{dst_cl}/aa_{instns}' if mode == 'train' else f'{ds}/{dst_cl}/zz_{instns}')
                    if mode == 'train':
                        new_n_train[src_cl] -= 1
                        new_n_train[dst_cl] += 1

for clss in r_clss[:-1]:
    data[clss] = os.listdir(f'{ds}/{clss}')

for clss in data:
    train_data = data[clss][:new_n_train[clss]]
    test_data = data[clss][new_n_train[clss]:]
    
    random.Random(0).shuffle(train_data)
    random.Random(0).shuffle(test_data)
    
    for i in range(len(data[clss])):
        if i < new_n_train[clss]:
            instns = train_data[i]
            os.rename(f'{ds}/{clss}/{instns}', f'{ds}/{clss}/train_{clss}_{i+1:04d}.txt')
        else:
            instns = test_data[i-new_n_train[clss]]
            os.rename(f'{ds}/{clss}/{instns}', f'{ds}/{clss}/test__{clss}_{i+1:04d}.txt')


for clss in r_clss[:-1]:
    data[clss] = os.listdir(f'{ds}/{clss}')

for clss in data:
    for name in data[clss]:
        os.rename(f'{ds}/{clss}/{name}', f'{ds}/{clss}/{name[6:]}')


print('Dataset was replaced')
