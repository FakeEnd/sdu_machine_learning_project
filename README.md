# sdu_machine_learning_project
This is the homework for the machinelearning in sdu


## deadline
实验得要在2022.2.7号结束
报告完成三天，最后提交要在2022.2.12号之前提交完成

## plan
我们计划采用多种机器学习的方法 + 多种深度学习的方法对项目进行实验和比较

### machine learning
+ [X] SVM  
+ [X] random forst
+ [X] KNN
+ [X] bayesian
+ [X] decision tree

### deep learning
+ [X] RNN
+ [X] GRU
+ [X] TextCNN
+ [X] Transformer
+ [X] BERT

## conclusion
我们最终选择了bert和TextCNN的融合模型，仅仅采用了name和address拼接后得到的序列来进行分类，最终的效果还较为满意，但是我觉得还有很多可以改进的方法，经纬度的坐标是可以利用的，如果和参考论文一样，采用基于distance的聚类分类可能效果会更好一些。其次不同因素的信息融合比较简单粗暴，只是将不同信息进行concat，在这里有很大的改进空间，尤其是在name和address，以及以后可以添加的经纬度三种因素的融合上，可以采取加权的方式，让不同的信息者更有侧重点，这样避免address的粗粒度干预到我们细粒度正常的分类中。

最后我想提及一下address这个维度的坐标真的有必要存在或者利用吗，诚然，加上可以，但是感觉不如用经纬度来提取更trandustive的信息，不仅让信息收集的困难度下降，也会一定程度的让模型对相同address却有同一分类的困惑度减少。
