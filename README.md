# BERT-DES
BERT预训练模型进行文本向量化表征，再采用动态集成选择算法完成分类器的构建与预测。
本文调用bert-based-service中的BertClient()类的encode()函数实现文本向量化。调用服务时，采用max_seq_len=512参数，其它参数默认。
本实验中代码及实验数据已上传，例如，1027bert_max_seq_len_512为影评数据集“Dennis+Schwartz”经过BERT预训练模型转化为向量的结果。
文件"desdcsdyn对比.py"中，定义了本研究采用的动态集成算法类，可直接调用。
同时，也上传了本研究的实验结果。
