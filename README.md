# SSDMV

The source code of the paper ''SSDMV: Semi-supervised Deep Social Spammer Detection by Multi-View Data Fusion''. 
The code is modified from code of ladder network:
    1) https://github.com/rinuboney/ladder
    2) https://github.com/rinuboney/ladder. 


Three .py files:
1) input_data.py: contains the functions to load data from files. function 'read_data_sets_views' load pre-learned representation vectors from different views. 
2) LadderClass.py: the model structure file;
3) TrainLadder.py: main entrance of the SSDMV

To run the model, just run python TrainLadder.py

Other notices:
1) open data sources: 
      1) Twitter: http://web.cs.wpi.edu/~kmlee/data.html;
      2) Sina: https://aminer.org/data-sna\#Weibo-Net-Tweet;
2) Embedding models used in data proprecessing part:
      1) Node2Vec: https://github.com/aditya-grover/node2vec
      2) Doc2Vec: https://radimrehurek.com/gensim/models/doc2vec.html
3) With the above tools and default parameters, it will be easy to generate prepocessed representation vectors of different views and reproduce the experimental results.
