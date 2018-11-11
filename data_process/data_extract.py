import pdb


data1 = open('nohup.out','r')
index = 1
scores = []
flag = 0
flag1 = 0
for line in data1.readlines():
    if line == 'model'+str(index)+'\n':
        scores.append([])
        index += 1
    if flag == 1:
        scores[index-2].append(line)
        flag = 0
    if line == 'max f1\n':
        flag = 1
data1.close()
pdb.set_trace()

