# Graph RAG

## 簡介
```
Graph-RAG 是將基礎的 RAG (向量資料庫) 改成圖資料庫為主的知識圖表，
簡單來說，就是把檢索分成2個階段: 
第一階段:
將參考文件以圖表的方式畫出來，並標記好所有的社群關係，
從(每個)社群關係中找出重點摘要
第二階段:將每個重點摘要統整出最後的答案

結論: 
Graph-RAG 在生成答案的多元性及完整性都高於基礎的 RAG 許多(因為有圖表檢索的幫助)
```

<div style="break-after: page; page-break-after: always;"></div> 

## (推理)流程
![Graph-RAG](https://hackmd.io/_uploads/S16y_zJ3C.png)  
### Source Documents → Text Chunks  
```
從參考文件中把與輸入問題相關的段落檢索出來，
並將檢索出來的段落分成多個文字區塊。
每一個文字塊會搭配一組提示詞彙,以抽取圖形索引的各種元素。

備註:
1. 較長的文字塊進行類別檢索時，需要啟用語言模型的次數並不多，
但會因為較長的文字區塊而導致語言模型的記憶力下降。
2. 較小的 chunk size 需要的實際參考值越高
3. 一般而言引用越多越好，但檢索過程需要平衡執行任務的呼叫次數與精確度。
```  
### Text Chunks → Element Instances  
```
從每段檢索出來的文字中識別並擷取圖形節點和邊緣的物件(使用 multipart LLM prompt 來完成)，
需要找出(物件名稱、類型、敘述、所有物件間的關聯性)，
來源實例(Instances)及目標實例(Instances)都會個別以(Tuple)的形式輸出
如果沒有擷取仔細或是有漏掉，
需要進行第二次檢索(還有一種特例，碰到專門領域的知識時)

備註:
1.遇到不同領域的文件需要訂製語料庫
2.多次檢索可以有效提升知識圖表的品質
3.多次檢索會使用到 Self-RAG 的概念(要求語言模型評估是否有漏掉的細節)
```  
<div style="break-after: page; page-break-after: always;"></div>

### Element Instances → Element Summaries  
```
抽象摘要: 使用 LLM 來「擷取」原始文字中所代表的實體、關係和主張的描述
將所有檢索出來的(實例摘要)轉換成圖表元素(多個) 的文字區塊(清晰描述)，
為了轉換成圖表元素(需要再進行一次 LLM 摘要，
利用 LLM 本身的邏輯來進行擷取重點並轉換成圖表元素)。
此步驟建立的索引類似(同質無向加權圖 undirected weighted graph)

同質無向加權圖 undirected weighted graph:
實體節點由關係邊連接，而邊的權重為偵測到的關係實例(利用標準化計數)

可能會出現的問題: 
LLM 可能無法以相同的文字格式持續擷取對同一個實體的引用，
造成重複的實體元素，進而造成實體圖中重複的節點。
不過，由於所有相關的實體「群組」都會在下一步中被偵測與總結，
加上 LLM 可以理解多重名稱變異背後的共同實體，
因此只要所有變異與相關實體的共用點有足夠的關聯性，
整體方法對於變異是有彈性的。
總而言之，
需要再可能會產生雜訊的圖表結構中，
使用豐富的描述性文字來處理同質節點，
這符合 LLM 的能力，以及全局性、以查詢為重點的總結需求。
這些特質也讓圖表索引有別於典型的知識圖表，
(典型的知識圖表)依賴簡潔一致的知識三元組(主體、謂語、客體)來進行推理任務。
```
### Element Summaries → Graph Communities  
```
使用 Leiden 演算法來將圖表劃分為多個群組(群組中的關係強度遠大於群組外部)，
以互斥(mutually exclusive)、集體窮盡(collective exhaustive)的方式涵蓋圖表的節點，
從而實現分而治之(divide and conquer)的全局總結(global summarization)
```  

<div style="break-after: page; page-break-after: always;"></div>

### Graph Communities → Community Summaries  
```
將資料集擴充至超大型資料集所設計的方法，
幫助 Leiden 層級架構中的每個群落建立重點摘要。
摘要很有用，可以了解資料集的整體結構和語意，
在沒有問題的情況下，摘要也可以用來釐清語料的意義，
把主要的重點放在回答全局查詢的圖形索引的其中一部分。

群落摘要 (Community Summaries) 的產生方式：
1. Leaf-level communities (樹葉型的群落): 
把 (Leaf-level communities) 的元素摘要
（nodes、edges、共變數(covariates)）排序，
然後反復加入 LLM 的 (context) ，直到達到符號限制為止。
優先順序：
每個社群邊緣 (community edge)，
依來源與目標節點合併程度的遞減順序，
加入來源節點、目標節點、連結共變數以及(edge)本身的描述。

2. Higher-level communities (高階的群落): 
如果所有元素摘要都在 (LLM contexts) 的符號限制內，
則採取與葉層群落相同的步驟，
(Community Summaries) 內的所有元素摘要 (element summaries)。
否則，會以 (element summaries) 符號遞減的順序排列子群落，
並以子群落摘要（較短）取代相關元素摘要（較長），
直到符合 (LLM contexts) 為止。
```  

<div style="break-after: page; page-break-after: always;"></div>

#### MultiHop-RAG 圖形群組索引(Graph communities)(使用 Leiden 演算法)  
![MultiHop-RAG 圖形群組索引](https://hackmd.io/_uploads/rkgEdfk2C.png)  

```
節點佈局透過 OpenORD 和 Force Atlas 2 (演算法)執行(節點顏色代表實體群落)
```  
- 分層聚類:  
    - Level 0:  
        ```
        具有最大模組化的層次分割(hierarchical partition with maximum modularity)
        ```  
    - Level 1:  
        ```
        顯示根部(底層)群落的內部結構(internal structure within root-level communities)
        ```

<div style="break-after: page; page-break-after: always;"></div> 

### Community Summaries → Community Answers → Global Answer  
```
社群摘要(上一步產生)用於在此階段產生最終答案(輸入:使用者提問)
社群結構的層級性質也意味著可以使用不同層級的社群摘要來回答問題，
這也提出了一個問題：
在層級性社群結構中，
某個特定層級是否提供了摘要細節與一般感知問題範圍的最佳平衡
```  
#### 對於特定的社群層級，答案產生方式  
- 準備社群摘要(Prepare community summaries):  
    ```
    社群摘要會隨機洗牌，並分割成預先指定代幣大小的區塊。
    這可確保相關資訊分佈在各個區塊中，
    而不是集中在單一上下文視窗中 (並可能遺失)。
    ```  
- 映射社群答案(Map community answers):  
    ```
    並行產生中間答案，每個分塊一個。
    LLM 也會被要求產生 0-100 分之間的分數，
    表示所產生的答案對回答目標問題的幫助程度。
    得分為 0 的答案會被篩選出來。
    ```  
- 還原為全局答案(Reduce to global answer):  
    ```
    中間的社群答案會依有用性分數從高到低排序，
    並反覆加入新的上下文視窗，直到達到代幣限制為止。
    這個最終的上下文會用來產生回傳給使用者的全局答案。
    ```  

<div style="break-after: page; page-break-after: always;"></div>

## Knowledge Graph
```
定義: 累積和傳達現實世界知識的資料圖形

Knowledge Graph (知識圖表): 
1. 儲存在圖資料庫 
2. 以圖描述實體 (物件、事件、情境、概念) 之間的關係
3. 儲存內容: 
文本資料(包含相互關係) + 組成原則(organizing principles)

Knowledge Graph 的組成:
Nodes: 表示相關的實體(the entities of interest)
Edges: 表示實體間的關係(relations between the entities)
```  
 
![Knowledge Graph](https://hackmd.io/_uploads/SyFI_zJ3C.png)

<div style="break-after: page; page-break-after: always;"></div>

### Key Characteristics(特性)  
#### Nodes  
```
儲存實體的相關資訊(人、地方、物件、機構等...)，
相比普通資料庫增加了可讀性及檢索的全面性
```

![Nodes](https://hackmd.io/_uploads/ByXFOGJ3C.png)

#### Relationships  
```
表示實體之間的關係
```  

![Relationships](https://hackmd.io/_uploads/ry_9lLQhR.png)

<div style="break-after: page; page-break-after: always;"></div> 

#### Organizing Principle:  
```
一種核心假設，
所有的東西都可以從這個核心假設中得到分類或價值(可提供靈活的概念性結構)
假設: 
一個中心參考點，讓其他物件都能被定位，通常用在概念架構中
```

![Organizing Principle](https://hackmd.io/_uploads/Hyea_G12R.png)  

##### Organizing:  
```
對特定領域的概念(包含概念之間關係)的正式說明
常見的表示方式:圖形網路表示
```  

<div style="break-after: page; page-break-after: always;"></div>

### Knowledge Graph 相關技術
#### Knowledge Graph Embedding
```
將知識圖形的實體和關係映射到低維向量空間，
以便有效地捕捉知識圖形的語義和結構，
得到的特徵向量可以用來訓練模型，
使其更能應用在人工智慧上
```  
##### Embedding methods
###### Tensor factorization-based
```
基於矩陣分解的方法通過從
使用者（user）和項目（item）中提取潛在因子（latent factor）
來調整總體評級以進行推薦
```  
![factorization](https://hackmd.io/_uploads/B1kQZL7n0.png)

<div style="break-after: page; page-break-after: always;"></div> 

公式:  
$$\underset{U,V}{argmin}=\sum_{i=0}^{m}\sum_{j=0}^{n}(D_{i,j}-\hat{D}_{i,j})^{2}$$  

$U_{i}$:   
```
Matrix U 第 i 行的向量。
```  

$V_{j}$:   
```
Matrix V 第 j 行的向量。
```   

$\hat{D}_{i,j}$:  

![f](lagrida_latex_editor.png)  


<div style="break-after: page; page-break-after: always;"></div>

###### Knowledge Acquisition(擷取)
```
著重於知識圖形的建模與建構，知識是透過映射語言從結構化的來源匯入，
也可以採用關係、實體或屬性萃取方法，
從非結構化文檔（例如新聞、研究論文等...）中萃取知識
```  
![Acquisition](https://hackmd.io/_uploads/BktWWIm20.png)

###### Knowledge Graph Completion
```
大多數的知識圖表仍然缺乏大量的實體和關係
使用2種方式來改善(知識圖表必須是靜態):
1.採用連結預測技術來產生三元組，然後給予三元組可信度分數
2.採用實體預測方法，從外部來源取得並整合進一步的資訊
```  

<div style="break-after: page; page-break-after: always;"></div> 

###### Knowledge Fusion  

```
定義: 將來自不同知識來源的知識整合，以提高系統的可解釋性和穩健性

可分為以下幾種類型:
1.規則知識融合(Rule-Based Knowledge Fusion, RBKF)：
將來自不同規則知識來源的知識進行融合，
以提高系統的可解釋性和穩健性
2.例子知識融合(Example-Based Knowledge Fusion, EBKF)：
將來自不同例子知識來源的知識進行融合，
以提高系統的可解釋性和穩健性
3.結構知識融合(Structural Knowledge Fusion, SKF)：
將來自不同結構知識來源的知識進行融合，
以提高系統的可解釋性和穩健性

核心原理與具體操作步驟:
1. 知識預處理：將不同知識來源的資訊清洗、轉換與整合
2. 知識擷取：從知識來源中提取有意義的知識，以便進行後續的知識融合
3. 知識融合：將不同知識來源的知識整合，以得到融合後的知識
4. 結果解釋：對融合後的知識進行解釋和分析，以提高系統的可解釋性和穩健性
```

公式:  
$$K = \frac{\sum{i=1}^{n} wi \cdot ki}{\sum{i=1}^{n} w_i}$$


$K$:   
```
融合後的知識
```  

$wi$:   
```
每個知識來源的權重
```  

$ki$:   
```
每個知識來源的知識
```

<div style="break-after: page; page-break-after: always;"></div> 

###### Knowledge Reasoning
```
定義: 根據現有資料推斷出新的事實

主要方法包括：
1. 基於邏輯規則的推理(logic rule-based)
2. 基於分散表示的推理(distributed representation-based methods)
3. 基於神經網絡的推理(neural network-based methods)
```  

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>