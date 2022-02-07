#predict_pos_list:n*1矩阵，n为测试集数据大小，存储测试集数据预测位置
#true_pos_list:n*1矩阵，n为测试集数据大小，存储测试集数据的真实位置

import heapq

def Find(list,x):
  for i,item in enumerate(list):
    if item==x:
        #print(i)
        return i
  else:
    return None

def top1_accuracy(predict_pos_list, true_pos_list):
    predict_pos_list = predict_pos_list.cpu().detach().numpy()
    true_pos_list = true_pos_list.cpu().detach().numpy()

    total_num = len(true_pos_list)
    accuracy = 0
    for i in range(total_num):
        temp = heapq.nlargest(1, predict_pos_list[i])
        top1_predict=[]
        top1_predict.append(Find(predict_pos_list[i],temp))

        if true_pos_list[i] in top1_predict:
            accuracy = accuracy + 1
    # top1_accuracy = accuracy / total_num
    return accuracy