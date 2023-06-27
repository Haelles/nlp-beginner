# nlp-beginner
* Researched and utilized preprocessing techniques and word embedding methods, including Bag-of-Words, N-gram, GloVe, Word2Vec, etc.
* Implemented models such as CNN, RNN, LSTM for tasks like sentiment classification, text matching, etc.


## Environment
### Version
torch 1.5.1

torchtext 0.6.0

torchvision 0.6.0

spacy 3.3.0

jsonlines


### 备注
在https://featurize.cn/ 服务器租借平台上测试，使用CUDA 11.2 + torch 1.10 + torchvision 0.11 + torchtext 0.6.0 + spacy 3.3.1也可以跑通


### 预下载
通过`python -m spacy download en_core_web_sm`下载en_core_web_sm

glove预训练权重下载-国内链接：https://blog.csdn.net/weixin_44912159/article/details/105538891?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-9-105538891-null-null.pc_agg_new_rank&utm_term=%E4%B8%AD%E6%96%87glove%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%90%91%E9%87%8F&spm=1000.2123.3001.4430

torchtext第一次根据glove权重构造Vector时候时间会比较长，可根据进度条等待


### torchtext ValueError
torchtext—ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
可参考https://blog.csdn.net/qq_23262411/article/details/100173224?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-100173224-blog-114952212.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-100173224-blog-114952212.pc_relevant_aa&utm_relevant_index=2

