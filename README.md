# Accepted-paper
Welcome to quote published articles, the code has been uploaded and updated.

## 论文1
### Title: A Novel Combined Prediction Scheme Based on CNN and LSTM for Urban PM<sub>2.5</sub> Concentration
* Aabstract: Urban air pollutant concentration prediction is dealing with a surge of massive environmental monitoring data and complex changes in air pollutants. This requires effective prediction methods to improve prediction accuracy and to prevent serious pollution incidents, thereby enhancing environmental management decision-making capacity. In this paper, a new pollutant concentration prediction method is proposed based on the vast amounts of environmental data and deep learning techniques. The proposed method integrates big data by using two kinds of deep networks. This method is based on the design that uses a convolutional neural network as the base layer, automatically extracting features of input data. A long short-term memory network is used for the output layer to consider the time dependence of pollutants. Our model consists of these two deep networks. With performance optimization, the model can predict future particulate matter (PM2.5) concentrations as a time series. Finally, the prediction results are compared with the results of numerical models. The applicability and advantages of the model are also analyzed. The experimental results show that it improves prediction performance compared with classic models.

* Latex inference:


     @article{qin2019novel,  
      title={A novel combined prediction scheme based on CNN and LSTM for urban PM 2.5 concentration},  
      author={Qin, Dongming and Yu, Jian and Zou, Guojian and Yong, Ruihan and Zhao, Qin and Zhang, Bo},  
      journal={IEEE Access},  
      volume={7},  
      pages={20050--20059},  
      year={2019},  
      publisher={IEEE}  
    }
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/A%20Novel%20Combined%20Prediction%20Scheme%20Based%20on%20CNN%20and%20LSTM%20for%20Urban%20PM2.5%20Concentration)

## 论文2
### A novel Encoder-Decoder model based on read-first LSTM for air pollutant prediction
* Aabstract: Accurate air pollutant prediction allows effective environment management to reduce the impact of pollution and prevent pollution incidents. Existing studies of air pollutant prediction are mostly interdisciplinary involving environmental science and computer science where the problem is formulated as time series prediction. A prevalent recent approach to time series prediction is the Encoder-Decoder model, which is based on recurrent neural networks (RNN) such as long short-term memory (LSTM), and great potential has been demonstrated. An LSTM network relies on various gate units, but in most existing studies the correlation between gate units is ignored. This correlation is important for establishing the relationship of the random variables in a time series as the stronger is this correlation, the stronger is the relationship between the random variables. In this paper we propose an improved LSTM, named Read-first LSTM or RLSTM for short, which is a more powerful temporal feature extractor than RNN, LSTM and Gated Recurrent Unit (GRU). RLSTM has some useful properties: (1) enables better store and remember capabilities in longer time series and (2) overcomes the problem of dependency between gate units. Since RLSTM is good at long term feature extraction, it is expected to perform well in time series prediction. Therefore, we use RLSTM as the Encoder and LSTM as the Decoder to build an Encoder-Decoder model (EDSModel) for pollutant prediction in this paper. Our experimental results show, for 1 to 24 h prediction, the proposed prediction model performed well with a root mean square error of 30.218. The effectiveness and superiority of RLSTM and the prediction model have been demonstrated.  

* Latex inference:


    @article{zhang2021novel,  
      title={A novel Encoder-Decoder model based on read-first LSTM for air pollutant prediction},  
      author={Zhang, Bo and Zou, Guojian and Qin, Dongming and Lu, Yunjie and Jin, Yupeng and Wang, Hui},  
      journal={Science of The Total Environment},  
      volume={765},  
      pages={144507},  
      year={2021},  
      publisher={Elsevier}  
    }
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/A%20novel%20Encoder-Decoder%20model%20based%20on%20read-first%20LSTM%20for%20air%20pollutant%20prediction)
* code link [click](https://github.com/zouguojian/Read-first-LSTM)

## 论文3
### RCL-Learning- ResNet and Convolutional Long Short-Term Memory-based Spatiotemporal Air Pollutant Concentration Prediction Model
* Aabstract: Predicting the concentration of air pollutants is an effective method for preventing pollution incidents by providing an early warning of harmful substances in the air. Accurate prediction of air pollutant concentration can more effectively control and prevent air pollution. In this study, a big data correlation principle and deep learning technology are used for a proposed model of predicting PM2.5 concentration. The model comprises a deep learning network model based on a residual neural network (ResNet) and a convolutional long short-term memory (LSTM) network (ConvLSTM). ResNet is used to deeply extract the spatial distribution features of pollutant concentration and meteorological data from multiple cities. The output is used as input to ConvLSTM, which further extracts the preliminary spatial distribution features extracted from the ResNet, while extracting the spatiotemporal features of the pollutant concentration and meteorological data. The model combines the two features to achieve a spatiotemporal correlation of feature sequences, thereby accurately predicting the future PM2.5 concentration of the target city for a period of time. Compared with other neural network models and traditional models, the proposed pollutant concentration prediction model improves the accuracy of predicting pollutant concentration. For 1- to 3-hours prediction tasks, the proposed pollutant concentration prediction model performed well and exhibited root mean square error (RMSE) between 5.478 and 13.622. In addition, we conducted multiscale predictions in the target city and achieved satisfactory performance, with the average RMSE value able to reach 22.927 even for 1- to 15-hours prediction tasks.  

* Latex inference:


    @article{zhang2022rcl,  
      title={RCL-Learning: ResNet and convolutional long short-term memory-based spatiotemporal air pollutant concentration prediction model},  
      author={Zhang, Bo and Zou, Guojian and Qin, Dongming and Ni, Qin and Mao, Hongwei and Li, Maozhen},  
      journal={Expert Systems with Applications},  
      pages={118017},  
      year={2022},  
      publisher={Elsevier}  
    }  
        
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/RCL-Learning-%20ResNet%20and%20Convolutional%20Long%20Short-Term%20Memory-based%20Spatiotemporal%20Air%20Pollutant%20Concentration%20Prediction%20Model)
* code link [click](https://github.com/zouguojian/RCL-Learning)
* code link [click](https://codeocean.com/capsule/6049117/tree)