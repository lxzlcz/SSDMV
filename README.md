# SSDMV

The source code of the paper ''SSDMV: Semi-supervised Deep Social Spammer Detection by Multi-View Data Fusion''. The code is modified from https://github.com/rinuboney/ladder. The code annotations has been added.

Three .py files:
1) input_data.py: contains the functions to load data from files. function 'read_data_sets_views' load pre-learned representation vectors from different views. 
2) LadderClass.py: the model structure file;
3) TrainLadder.py: main entrance of the SSDMV

To run the model, just run python TrainLadder.py

Other notices:
1) data sources: 
      Twitter: http://web.cs.wpi.edu/~kmlee/data.html 
      Sina: https://aminer.org/data-sna\#Weibo-Net-Tweet
2) Embedding models used in data proprecessing part:
      Node2Vec: https://github.com/aditya-grover/node2vec
      Doc2Vec: https://radimrehurek.com/gensim/models/doc2vec.html
3) Due to the upload size limitation of github, we cannot upload the pre-learned representation vectors of different views. However with the above tools and default parameters, it will be easy to generate representations and reproduce the experomental results.
