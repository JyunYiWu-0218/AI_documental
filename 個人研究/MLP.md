# MLP
```
結構: 輸入(Input)層 --> 隱藏(Hidden)層 --> 輸出(Output)層
定義: 前向傳遞類神經網路
```  
![MLP](images/MLP.png)  

<div style="break-after: page; page-break-after: always;"></div> 

## Input layer --> Hidden layer  
### Input layer --> Hidden layer input 

$$H_{j}=\sum_{i=0}^{d}v_{ij}x_{i}$$  

- $v_{ij}$:  
    ```
    第i個輸入到第j個hidden node的權重
    ```  

- Hidden layer input --> Hidden layer output:  
    ```
    經過激活函數產生 Hidden layer output
    ```  

    $$H_{o}=activation(H_{j})$$  
        
### Hidden layer --> Output layer   
- Hidden layer --> Output layer input:  

    $$O_{j}=\sum_{o=0}^{p}w_{oi}H_{o}$$  

- Output layer input --> Output layer output(y):  

    ![Output_layer](xxx.png)  

<div style="break-after: page; page-break-after: always;"></div> 

### 反向傳遞 (Backward propagation)  
```
利用反向傳遞進行參數更新(直到誤差收斂)
```  
#### Output --> Hidden layer output  
```
偏微分方程
```  

$$\frac{\partial E^{(i)}}{\partial w_{oi}} = \frac{\partial E^{(i)}}{\partial O_{j}}*\frac{\partial O_{j}}{\partial w_{oi}}$$  

$\frac{\partial E^{(i)}}{\partial O_{j}}$  

![Hidden](bbb.png)  

$\frac{\partial O_{j}}{\partial w_{oi}}$  

$$\frac{\partial O_{j}}{\partial w_{oi}}=\frac{\partial \sum_{i=0}^{q}w_{oi}H_{o}}{\partial w_{oi}}=H_{o}$$  


<div style="break-after: page; page-break-after: always;"></div> 

#### Hidden layer --> Input layer  

![Hidden_Input](ccc.png)  

- $\frac{\partial E^{(i)}}{\partial O_{j}}$:  
![input](cdc.png)  

- $\frac{\partial O_{j}}{\partial H_{j}}$:  
$$\frac{\partial O_{j}}{\partial H_{j}}=w_{oi}activation\;function^{'}(H_{j}^{(i)})$$  

- $\frac{\partial H_{j}}{\partial v_{ij}}$:  
$$\frac{\partial \sum_{i=0}^{d}v_{ij}x_{i}}{\partial v_{ij}}=x_{i}$$  


<div style="break-after: page; page-break-after: always;"></div> 

#### 最佳參數解  
```
利用反向傳遞 (Backward propagation) 找出
```  
##### $v_{ij}$  
$$v_{ij}=v_{ij}-\eta\Delta v_{ij}$$  

###### $\Delta v_{ij}$  
![g](nbn.png)  

##### $w_{oi}$  

$$w_{oi}=w_{oi}-\eta\Delta w_{oi}$$  

###### $\Delta w_{oi}$  

![w](w.png)  


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> 
MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>