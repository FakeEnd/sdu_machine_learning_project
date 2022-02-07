import heapq

def Find(list,x):
  for i,item in enumerate(list):
    if item==x:
        #print(i)
        return i
  else:
    return None

def top5_accuracy(predict_pos_list, true_pos_list):
    predict_pos_list = predict_pos_list.cpu().detach().numpy()
    true_pos_list = true_pos_list.cpu().detach().numpy()

    total_num = len(true_pos_list)
    accuracy = 0
    for i in range(total_num):
        temp = heapq.nlargest(5, predict_pos_list[i])
        top5_predict=[]
        for j in range(len(temp)):
            top5_predict.append(Find(predict_pos_list[i],temp[j]))

        if true_pos_list[i] in top5_predict:
            accuracy = accuracy + 1;
    # top5_accuracy = accuracy / total_num
    return accuracy