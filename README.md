# Accepted-paper
Welcome to quote our published papers, the codes have been uploaded. If you have any questions, do not hesitate to contact us.  
Our e-mail address is: 2010768@tongji.edu.cn;  
Our WeChat ID is: GuoJianZou;  
Thanks !!!!!!  

<div align=center><img src ="https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fkk.51.com%2Frefer%3Furl%3Dhttps%3A%2F%2Fmmbiz.qpic.cn%2Fmmbiz_gif%2FzxqIk8VJIrykWy7eNkw8aNBou0XrWmX585bD3dsClf8BUWic2XRZl4kaND8XJHBQn8LOI83VzAdwvI7fXTypxrQ%2F0%3Fwx_fmt%3Dgif&refer=http%3A%2F%2Fkk.51.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1661072003&t=e7891ac13714e529265e8d6ad640a55d"/></div>

<!-- ![paper](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fkk.51.com%2Frefer%3Furl%3Dhttps%3A%2F%2Fmmbiz.qpic.cn%2Fmmbiz_gif%2FzxqIk8VJIrykWy7eNkw8aNBou0XrWmX585bD3dsClf8BUWic2XRZl4kaND8XJHBQn8LOI83VzAdwvI7fXTypxrQ%2F0%3Fwx_fmt%3Dgif&refer=http%3A%2F%2Fkk.51.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1661072003&t=e7891ac13714e529265e8d6ad640a55d) -->

## 2023年
### When Will We Arrive? A Novel Multi-Task Spatio-Temporal Attention Network Based on Individual Preference for Estimating Travel Time, IEEE Transactions on Intelligent Transportation Systems, 1558-0016. (SCI, 中科院一区, IF: 9.551, <font color=#FF000>co-first author</font>)  

<div align=center><img src ="https://github.com/zouguojian/Accepted-paper/blob/main/RCL-Learning%20ResNet%20and%20Convolutional%20Long%20Short-Term%20Memory-based%20Spatiotemporal%20Air%20Pollutant%20Concentration%20Prediction%20Model/image/figure5.png"/></div>

* Abstract: 
> Predicting how long a trip will take may allow travelers plan ahead, save money, and avoid traffic congestion. The journey time estimation model should take into account three crucial factors: (1) individual travel preference, (2) dynamic spatio-temporal correlations, and (3) the association between long-term speed forecast and travel time estimate. In order to overcome these challenges, this study proposes a unique parallel architecture called the multi-task spatio-temporal attention network (MT-STAN) to estimate journey times. To extract the dynamic spatio-temporal correlations of the road network, we first develop a traffic speed prediction model based on spatio-temporal block and bridge transformer networks, combining the road, timestamp, and traffic speed information into hidden states. Second, we offer a personalized model for estimating journey times that makes use of cross-network, holistic attention, and semantic transformer. In this approach, travel preferences extraction through cross-network, holistic attention permits correlations between the dynamic road network’s hidden states and individual journey characteristics, which are subsequently transformed into global semantics by the semantic transformer; preferences and semantics are integrated during the estimate phase. Finally, a multi-task learning component is included, which combines both traffic speed prediction and individual journey time estimate, via the sharing of underlying network parameters and the improvement of the contextual semantic knowledge of the latter job. Evaluation experiments are carried out using a highway dataset collected in Yinchuan City, Ningxia Province, China. The proposed prediction model outperforms state-of-the-art baseline approaches in experiments.   

* Latex inference:

    @ARTICLE{10133870,
        author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Tu, Meiting and Fan, Jing and Li, Ye},
        journal={IEEE Transactions on Intelligent Transportation Systems}, 
        title={When Will We Arrive? A Novel Multi-Task Spatio-Temporal Attention Network Based on Individual Preference for Estimating Travel Time}, 
        year={2023},
        volume={},
        number={},
        pages={1-15},
        doi={10.1109/TITS.2023.3276916}
      }
        
* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/RCL-Learning-%20ResNet%20and%20Convolutional%20Long%20Short-Term%20Memory-based%20Spatiotemporal%20Air%20Pollutant%20Concentration%20Prediction%20Model)
* code link [click](https://github.com/zouguojian/Travel-time-prediction)

### A spatial correlation prediction model of urban PM2.5 concentration based on deconvolution and LSTM.

* Abstract: 
> Precise prediction of air pollutants can effectively reduce the occurrence of heavy pollution incidents. With the current surge of massive data, deep learning appears to be a promising technique to achieve dynamic prediction of air pollutant concentration from both the spatial and temporal dimensions. This paper presents Dev-LSTM, a prediction model building on deconvolution and LSTM. The novelty of Dev-LSTM lies in its capability to fully extract the spatial feature correlation of air pollutant concentration data, preventing the excessive loss of information caused by traditional convolution. At the same time, the feature associations in the time dimension are mined to produce accurate prediction results. Experimental results show that Dev-LSTM outperforms traditional prediction models on a variety of indicators.
* paper link [click](https://github.com/zouguojian/Accepted-paper/blob/main/A%20spatial%20correlation%20prediction%20model%20of%20urban%20PM2.5%20concentration%20based%20on%20deconvolution%20and%20LSTM/NEUCOM-D-22-06372.pdf)

* Latex inference:

    @article{li4342073spatial,  
      title={A Spatial Correlation Prediction Model of Urban Pm2. 5 Concentration Based on Deconvolution and Lstm},  
      author={Li, Maozhen and Zhang, Bo and Liu, Yuan and Yong, Ruihan and Zou, Guojian and Yang, Ru},  
      journal={Available at SSRN 4342073}  
    }  

## 2022年
### Deep learning for Air Pollutant Concentration Prediction: A Review, Atmospheric Environment, accept. (SCI, 中科院一区, IF: 5.755, <font color=#FF000>corresponding author</font>) 

<div align=center><img src ="https://github.com/zouguojian/Accepted-paper/blob/main/Deep%20learning%20for%20Air%20Pollutant%20Concentration%20Prediction-A%20Review/image/WechatIMG97.jpeg" width = "600" height="500"/></div>

* Abstract: 
> Air pollution has become one of the critical environmental problem in the 21st century and has attracted worldwide attentions. To mitigate it, many researchers have investigated the issue and attempted to accurately predict air pollutant concentrations using various methods. Currently, deep learning methods are the most prevailing ones. In this paper, we extend a comprehensive review on deep learning methods specifically for air pollutant concentration prediction. We start from the analysis on non-deep learning methods applied in air pollutant concentration prediction in terms of expertise, applications and deficiencies. Then, we investigate current deep learning methods for air pollutant concentration prediction from the perspectives of temporal, spatial and spatio-temporal correlations these methods could model. Further, we list some public datasets and auxiliary features used in air pollutant prediction, and compare representative experiments on these datasets. From the comparison, we draw some conclusions. Finally, we identify current limitations and future research directions of deep learning methods for air pollutant concentration prediction. The review may inspire researchers and to a certain extent promote the development of deep learning in air pollutant concentration prediction.
* paper link [click](https://github.com/zouguojian/Accepted-paper/blob/main/Deep%20learning%20for%20Air%20Pollutant%20Concentration%20Prediction-A%20Review/manuscript(clean).pdf)

* Latex inference:

    @article{zhang2022deep,  
      title={Deep learning for air pollutant concentration prediction: A review},  
      author={Zhang, Bo and Rong, Yi and Yong, Ruihan and Qin, Dongming and Li, Maozhen and Zou, Guojian and Pan, Jianguo},  
      journal={Atmospheric Environment},  
      pages={119347},  
      year={2022},  
      publisher={Elsevier}  
    }  


### RCL-Learning- ResNet and Convolutional Long Short-Term Memory-based Spatiotemporal Air Pollutant Concentration Prediction Model, Expert Systems With Applications, 118017. (SCI, 中科院一区, IF: 8.665, <font color=#FF000>co-first author</font>)  

<div align=center><img src ="https://github.com/zouguojian/Accepted-paper/blob/main/RCL-Learning%20ResNet%20and%20Convolutional%20Long%20Short-Term%20Memory-based%20Spatiotemporal%20Air%20Pollutant%20Concentration%20Prediction%20Model/image/figure5.png"/></div>

* Abstract: 
> Predicting the concentration of air pollutants is an effective method for preventing pollution incidents by providing an early warning of harmful substances in the air. Accurate prediction of air pollutant concentration can more effectively control and prevent air pollution. In this study, a big data correlation principle and deep learning technology are used for a proposed model of predicting PM2.5 concentration. The model comprises a deep learning network model based on a residual neural network (ResNet) and a convolutional long short-term memory (LSTM) network (ConvLSTM). ResNet is used to deeply extract the spatial distribution features of pollutant concentration and meteorological data from multiple cities. The output is used as input to ConvLSTM, which further extracts the preliminary spatial distribution features extracted from the ResNet, while extracting the spatiotemporal features of the pollutant concentration and meteorological data. The model combines the two features to achieve a spatiotemporal correlation of feature sequences, thereby accurately predicting the future PM2.5 concentration of the target city for a period of time. Compared with other neural network models and traditional models, the proposed pollutant concentration prediction model improves the accuracy of predicting pollutant concentration. For 1- to 3-hours prediction tasks, the proposed pollutant concentration prediction model performed well and exhibited root mean square error (RMSE) between 5.478 and 13.622. In addition, we conducted multiscale predictions in the target city and achieved satisfactory performance, with the average RMSE value able to reach 22.927 even for 1- to 15-hours prediction tasks.  

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

### Exploring the nonlinear impact of air pollution on housing prices-A machine learning approach, Economics of Transportation, 100272. (SCI, 中科院三区, IF: 2.829, **first author**)  

* Abstract:
> Air pollution has profoundly impacted residents’ lifestyles as well as their willingness to pay for real estate. Exploring the relationship between air pollution and housing prices has become increasingly prominent. Current research on housing prices mainly uses the hedonic pricing model and the spatial econometric model, which are both linear methods. However, it is difficult to use these methods to model the nonlinear relationship between housing price and its determinants. In addition, most of the existing studies neglect the effects of multiple pollutants on housing prices. To fill these gaps, this study uses a machine learning approach, the gradient boosting decision tree (GBDT) model to analyze the nonlinear impacts of air pollution and the built environment on housing prices in Shanghai. The experimental results show that the GBDT can better fit the nonlinear relationship between housing prices and various explanatory variables compared with traditional linear models. Furthermore, the relative importance rankings of the built environment and air pollution variables are analyzed based on the GBDT model. It indicates that built environment variables contribute 97.21% of the influences on housing prices, whereas the contribution of air pollution variables is 2.79%. Although the impact of air pollution is relatively small, the marginal willingness of residents to pay for clean air is significant. With an improvement of 1 g/m in the average concentrations of PM2.5 and NO, the average housing price increases by 155.93 Yuan/m and 278.03 Yuan/m, respectively. Therefore, this study can improve our understanding of the nonlinear impact of air pollution on housing prices and provide a basis for formulating and revising policies related to housing prices.  

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

### ST-ANet: Speed prediction of spatio-temporal attention network for dynamic highway network. (中文核心, 已收录, **first author**)  

* Abstract:
> Accurate prediction of highway traffic speed can reduce traffic accidents and transit time, provide valuable reference data for traffic control in advance, and is of great significance to highway management. This paper proposes a speed prediction of spatio-temporal attention network for dynamic highway network (ST-ANet) driven by data and long-term prediction tasks. ST-ANet consists of three parts, the graph attention network (GAN) based on the spatial attention mechanism, the temporal attention network based on the multi-head self-attention mechanism, and the long short-term memory network (LSTM). In addition, this paper also uses some techniques to improve model performance, including dense connections and layer batch normalization methods. First, use GAN to extract the dynamic spatial correlation features of the highway network; second, use LSTM to extract the temporal correlation features of the highway network; finally, use the temporal attention mechanism to focus on the correlation between historical input data and predicted values. The experimental verification uses the monitoring data of the highway network in Yinchuan City, Ningxia Province, China. Compared with the latest GCN-LSTM model, the ST-ANet model reduces the prediction error MAE of the highway network traffic speed in the next 1 hour, 2 hours, and 3 hours by 4.0%, 3.6% and 3.9%, respectively. The experimental results show that the performance of the proposed prediction model in this paper is better than the baseline methods and can effectively solve the problem of long-term highway network traffic speed prediction.  

* paper link [click](https://github.com/zouguojian/Accepted-paper/tree/main/ST-ANet%2C%20traffic%20speed%20prediction)

## 2021年
### A novel Encoder-Decoder model based on read-first LSTM for air pollutant prediction, Science of The Total Environment, 765, 144507. (SCI, 中科院一区, IF: 10.753, **co-first author**)  

<div align=center><img src ="https://github.com/zouguojian/Accepted-paper/blob/main/A%20novel%20Encoder-Decoder%20model%20based%20on%20read-first%20LSTM%20for%20air%20pollutant%20prediction/image/RLSTM-LSTM.png"/></div>

* Abstract:
> Accurate air pollutant prediction allows effective environment management to reduce the impact of pollution and prevent pollution incidents. Existing studies of air pollutant prediction are mostly interdisciplinary involving environmental science and computer science where the problem is formulated as time series prediction. A prevalent recent approach to time series prediction is the Encoder-Decoder model, which is based on recurrent neural networks (RNN) such as long short-term memory (LSTM), and great potential has been demonstrated. An LSTM network relies on various gate units, but in most existing studies the correlation between gate units is ignored. This correlation is important for establishing the relationship of the random variables in a time series as the stronger is this correlation, the stronger is the relationship between the random variables. In this paper we propose an improved LSTM, named Read-first LSTM or RLSTM for short, which is a more powerful temporal feature extractor than RNN, LSTM and Gated Recurrent Unit (GRU). RLSTM has some useful properties: (1) enables better store and remember capabilities in longer time series and (2) overcomes the problem of dependency between gate units. Since RLSTM is good at long term feature extraction, it is expected to perform well in time series prediction. Therefore, we use RLSTM as the Encoder and LSTM as the Decoder to build an Encoder-Decoder model (EDSModel) for pollutant prediction in this paper. Our experimental results show, for 1 to 24 h prediction, the proposed prediction model performed well with a root mean square error of 30.218. The effectiveness and superiority of RLSTM and the prediction model have been demonstrated.  

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

### FDN-learning- Urban PM2.5-concentration Spatial Correlation Prediction Model Based on Fusion Deep Neural Network, Big Data Research, 26, 100269. (SCI, 中科院三区, IF:3.739, **first author**) 

<div align=center><img src ="https://github.com/zouguojian/Accepted-paper/blob/main/FDN-learning%20Urban%20PM2.5-concentration%20Spatial%20Correlation%20Prediction%20Model%20Based%20on%20Fusion%20Deep%20Neural%20Network/image/figure4.png"/></div>

* Abstract:
> The problem of increasing air pollution poses a challenge to smart city development, as spatial air pollution correlation exists among adjacent cities. However, it is difficult to predict the degree of air pollution of a location by exploiting massive air pollution datasets incorporating data on spatially related locations. Construction of a spatial correlation prediction model for air pollution is therefore required for air pollution big-data mining. In this paper, we propose an air pollution-concentration spatial correlation prediction model based on a fusion deep neural network called FDN-learning. Three models are combined: a stacked anti-autoencoder network, Gaussian function model, and long short-term memory network (LSTM). The FDN-learning model is composed of three layers for feature expansion, intermediate processing, and data prediction. In the first layer, we employ a stacked anti-autoencoder model to learn the source-data spatial features through a feature expansion hidden layer; this can enrich the feature vector and mine more information for further prediction. In the second layer, the Gaussian function evaluates effective weights for the outputs of the stacked anti-autoencoder models in the preceding layer; the spatial correction effects are therefore incorporated in this layer. Finally, the LSTM model in the data prediction layer learns the air pollution-concentration temporal features. A fine-tuning method based on stochastic gradient descent is applied to the FDN-learning model for improved performance. Empirical results are used to verify the feasibility and effectiveness of our proposed model based on a real-world air pollution dataset.  

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

### Longer Time Span Air Pollution Prediction: The Attention and Autoencoder Hybrid Learning Model, Mathematical Problems in Engineering, 2021. (SCI, 中科院四区)
* Abstract:
> Air pollution has become a critical issue in human’s life. Predicting the changing trends of air pollutants would be of great help for public health and natural environments. Current methods focus on the prediction accuracy and retain the forecasting time span within 12 hours. Shorter time span decreases the practicability of these perditions, even with higher accuracy. This study proposes an attention and autoencoder (A&A) hybrid learning approach to obtain a longer period of air pollution changing trends while holding the same high accuracy. Since pollutant concentration forecast highly relates to time changing, quite different from normal prediction problems like autotranslation, we integrate “time decay factor” into the traditional attention mechanism. The time decay factor can alleviate the impact of the value observed from a longer time before while increasing the impact of the value from a closer time point. We also utilize the hidden states in the decoder to build connection between history values and current ones. Thus, the proposed model can extract the changing trend of a longer history time span while coping with abrupt changes within a shorter time span. A set of experiments demonstrate that the A&A learning approach can obtain the changing trend of air pollutants, like PM2.5, during a longer time span of 12, 24, or even 48 hours. The approach is also tested under different pollutant concentrations and different periods and the results validate its robustness and generality.  

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

## 2021以前
### Title: A Novel Combined Prediction Scheme Based on CNN and LSTM for Urban PM<sub>2.5</sub> Concentration, (SCI)
* Abstract:
> Urban air pollutant concentration prediction is dealing with a surge of massive environmental monitoring data and complex changes in air pollutants. This requires effective prediction methods to improve prediction accuracy and to prevent serious pollution incidents, thereby enhancing environmental management decision-making capacity. In this paper, a new pollutant concentration prediction method is proposed based on the vast amounts of environmental data and deep learning techniques. The proposed method integrates big data by using two kinds of deep networks. This method is based on the design that uses a convolutional neural network as the base layer, automatically extracting features of input data. A long short-term memory network is used for the output layer to consider the time dependence of pollutants. Our model consists of these two deep networks. With performance optimization, the model can predict future particulate matter (PM2.5) concentrations as a time series. Finally, the prediction results are compared with the results of numerical models. The applicability and advantages of the model are also analyzed. The experimental results show that it improves prediction performance compared with classic models.

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

### A Space-time Dimension User Preference Calculation Method for Recommendation in Social network, (EI)
* Abstract:
> Under the background of the Mobile Internet Age, location service has been developed rapidly. On the basis of modeling the space-time dimension and the study of users' personalized preference combined with the preference of similar user groups, this paper proposes the information selection model of location service and relevant algorithms. In this model, a space-time dimensional model is firstly constructed to process the information of users' personalized location service in the time and spatial dimension. Then, a new user preference model is constructed based on the existing study on user preference.  


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