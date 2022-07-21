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

## 论文4
### FDN-learning- Urban PM2.5-concentration Spatial Correlation Prediction Model Based on Fusion Deep Neural Network
* Aabstract: The problem of increasing air pollution poses a challenge to smart city development, as spatial air pollution correlation exists among adjacent cities. However, it is difficult to predict the degree of air pollution of a location by exploiting massive air pollution datasets incorporating data on spatially related locations. Construction of a spatial correlation prediction model for air pollution is therefore required for air pollution big-data mining. In this paper, we propose an air pollution-concentration spatial correlation prediction model based on a fusion deep neural network called FDN-learning. Three models are combined: a stacked anti-autoencoder network, Gaussian function model, and long short-term memory network (LSTM). The FDN-learning model is composed of three layers for feature expansion, intermediate processing, and data prediction. In the first layer, we employ a stacked anti-autoencoder model to learn the source-data spatial features through a feature expansion hidden layer; this can enrich the feature vector and mine more information for further prediction. In the second layer, the Gaussian function evaluates effective weights for the outputs of the stacked anti-autoencoder models in the preceding layer; the spatial correction effects are therefore incorporated in this layer. Finally, the LSTM model in the data prediction layer learns the air pollution-concentration temporal features. A fine-tuning method based on stochastic gradient descent is applied to the FDN-learning model for improved performance. Empirical results are used to verify the feasibility and effectiveness of our proposed model based on a real-world air pollution dataset.  

* Latex inference:


    @article{zou2021fdn,  
      title={FDN-learning: Urban PM2. 5-concentration Spatial Correlation Prediction Model Based on Fusion Deep Neural Network},  
      author={Zou, Guojian and Zhang, Bo and Yong, Ruihan and Qin, Dongming and Zhao, Qin},  
      journal={Big Data Research},  
      volume={26},  
      pages={100269},  
      year={2021},  
      publisher={Elsevier}  
    }  
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/FDN-learning-%20Urban%20PM2.5-concentration%20Spatial%20Correlation%20Prediction%20Model%20Based%20on%20Fusion%20Deep%20Neural%20Network)
* code link [click](https://github.com/zouguojian/FDN-Learning)

## 论文5
### Exploring the nonlinear impact of air pollution on housing prices- A machine learning approach
* Air pollution has profoundly impacted residents’ lifestyles as well as their willingness to pay for real estate. Exploring the relationship between air pollution and housing prices has become increasingly prominent. Current research on housing prices mainly uses the hedonic pricing model and the spatial econometric model, which are both linear methods. However, it is difficult to use these methods to model the nonlinear relationship between housing price and its determinants. In addition, most of the existing studies neglect the effects of multiple pollutants on housing prices. To fill these gaps, this study uses a machine learning approach, the gradient boosting decision tree (GBDT) model to analyze the nonlinear impacts of air pollution and the built environment on housing prices in Shanghai. The experimental results show that the GBDT can better fit the nonlinear relationship between housing prices and various explanatory variables compared with traditional linear models. Furthermore, the relative importance rankings of the built environment and air pollution variables are analyzed based on the GBDT model. It indicates that built environment variables contribute 97.21% of the influences on housing prices, whereas the contribution of air pollution variables is 2.79%. Although the impact of air pollution is relatively small, the marginal willingness of residents to pay for clean air is significant. With an improvement of 1 g/m in the average concentrations of PM2.5 and NO, the average housing price increases by 155.93 Yuan/m and 278.03 Yuan/m, respectively. Therefore, this study can improve our understanding of the nonlinear impact of air pollution on housing prices and provide a basis for formulating and revising policies related to housing prices.  

* Latex inference:


    @article{zou2022exploring,  
      title={Exploring the nonlinear impact of air pollution on housing prices: A machine learning approach},  
      author={Zou, Guojian and Lai, Ziliang and Li, Ye and Liu, Xinghua and Li, Wenxiang},  
      journal={Economics of Transportation},  
      volume={31},  
      pages={100272},  
      year={2022},  
      publisher={Elsevier}  
    }  
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/Exploring%20the%20nonlinear%20impact%20of%20air%20pollution%20on%20housing%20prices-%20A%20machine%20learning%20approach)


## 论文6
### A Space-time Dimension User Preference Calculation Method for Recommendation in Social network
* Aabstract: Under the background of the Mobile Internet Age, location service has been developed rapidly. On the basis of modeling the space-time dimension and the study of users' personalized preference combined with the preference of similar user groups, this paper proposes the information selection model of location service and relevant algorithms. In this model, a space-time dimensional model is firstly constructed to process the information of users' personalized location service in the time and spatial dimension. Then, a new user preference model is constructed based on the existing study on user preference.  


* Latex inference:


    @inproceedings{guojian2018space,  
      title={A space-time dimension user preference calculation method for recommendation in social network},  
      author={Guojian, Zou and Jisheng, Wang and Hailei, Yuan and Dong, Wang and Tao, Pan and Feng, Song and Bo, Zhang},  
      booktitle={2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA)},  
      pages={1643--1648},  
      year={2018},  
      organization={IEEE}  
    }   
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/A%20Space-time%20Dimension%20User%20Preference%20Calculation%20Method%20for%20Recommendation%20in%20Social%20network)

## 论文7
### Longer Time Span Air Pollution Prediction: The Attention and Autoencoder Hybrid Learning Model
* Aabstract: Air pollution has become a critical issue in human’s life. Predicting the changing trends of air pollutants would be of great help for public health and natural environments. Current methods focus on the prediction accuracy and retain the forecasting time span within 12 hours. Shorter time span decreases the practicability of these perditions, even with higher accuracy. This study proposes an attention and autoencoder (A&A) hybrid learning approach to obtain a longer period of air pollution changing trends while holding the same high accuracy. Since pollutant concentration forecast highly relates to time changing, quite different from normal prediction problems like autotranslation, we integrate “time decay factor” into the traditional attention mechanism. The time decay factor can alleviate the impact of the value observed from a longer time before while increasing the impact of the value from a closer time point. We also utilize the hidden states in the decoder to build connection between history values and current ones. Thus, the proposed model can extract the changing trend of a longer history time span while coping with abrupt changes within a shorter time span. A set of experiments demonstrate that the A&A learning approach can obtain the changing trend of air pollutants, like PM2.5, during a longer time span of 12, 24, or even 48 hours. The approach is also tested under different pollutant concentrations and different periods and the results validate its robustness and generality.  

* Latex inference:

    
    @article{tu2021longer,  
      title={Longer Time Span Air Pollution Prediction: The Attention and Autoencoder Hybrid Learning Model},  
      author={Tu, Xin-Yu and Zhang, Bo and Jin, Yu-Peng and Zou, Guo-Jian and Pan, Jian-Guo and Li, Mao-Zhen},  
      journal={Mathematical Problems in Engineering},  
      volume={2021},  
      year={2021},  
      publisher={Hindawi}  
    }   
    
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/Longer%20Time-Span%20Air%20Pollution%20Predictionl)