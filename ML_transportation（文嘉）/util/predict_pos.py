import heapq
import pickle
import csv

def Find(list,x):
  for i,item in enumerate(list):
    if item==x:
        #print(i)
        return i
  else:
    return None

def predict_pos(logits):
    with open('../data/unique_pos_list (1).pkl', 'rb') as fr:
        unique_pos_list = pickle.load(fr)

        fr.close()

    logits=logits.cpu().numpy()
    total_num=len(logits)
    predict_pos=[]
    for i in range(total_num):
        temp = heapq.nlargest(5, logits[i])
        top5_index=[]
        for j in range(len(temp)):
            location=unique_pos_list[Find(logits[i],temp[j])-4]
            top5_index.append(location)

        predict_pos.append(top5_index)

    #保存到csv中
    f = open('predict_top5_pos.csv', 'a+', newline='')
    writer = csv.writer(f)
    for i in predict_pos:
        writer.writerow(i)
    f.close()