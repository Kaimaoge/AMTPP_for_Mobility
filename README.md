# AMTPP_for_Mobility
[Individual Mobility Prediction via Attentive Marked Temporal Point Processes](https://arxiv.org/pdf/2109.02715.pdf) 

## Abstract
Individual mobility prediction is an essential task for transportation demand management and traffic system operation. There exists a large body of works on modeling location sequence and predicting the next location of users; however, little attention is paid to the prediction of the next trip, which is governed by the strong spatiotemporal dependencies between diverse attributes, including trip start time t, origin o, and destination d. To fill this gap, in this paper we propose a novel point process-based model---Attentive Marked Temporal Point Processes (AMTPP)---to model human mobility and predict the whole trip (t,o,d) in a joint manner. To encode the influence of history trips, AMTPP employs the self-attention mechanism with a carefully designed positional embedding to capture the daily/weekly periodicity and regularity in individual travel behavior. Given the unique peaked nature of inter-event time in human behavior, we use an asymmetric log-Laplace mixture distribution to precisely model the distribution of trip start time t. Furthermore, an origin-destination (OD) matrix learning block is developed to model the relationship between every origin and destination pair. Experimental results on two large metro trip datasets demonstrate the superior performance of AMTPP.

## Structure

<img src="https://github.com/Kaimaoge/AMTPP_for_Mobility/blob/main/fig/fig2-encoder2-1.png" width="800">
<img src="https://github.com/Kaimaoge/AMTPP_for_Mobility/blob/main/fig/fig2-emission2-1.png" width="800">

## Datasets
The Hangzhou mobility dataset can be downloaded from [TianChi](https://www.kaggle.com/zjplab/hangzhou-metro-traffic-prediction/activity). We use hangzhou_traj.py to obtain datasets for training and evaluation
