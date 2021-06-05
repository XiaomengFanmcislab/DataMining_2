#!/usr/bin/env python
# coding: utf-8

# # 红酒数据集频繁模式与关联规则挖掘
# ### 姓名：范啸猛
# ### 学号：3220200870
# ## 1. 数据集预处理
# 

# 红酒数据集中有11种属性，其中description，Unnamed: 0，designation与winery为unique的属性无法进行频繁模式挖掘，因此需要将上述属性删除。而国家，省份，区域存在较大的冗余与常识关系，因此在这里我们选取US国家作为研究对象，只保留省份属性，将非US国家的信息删除，并将属性'region_2', 'winery'删除。具体步骤如下：
# 首先，我们将无法进行频繁模式挖掘的列删除，即将'Unnamed: 0', 'description', 'designation', 'region_1', 'region_2', 'winery'列删除。处理数据的代码如下：

# In[207]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\winemag-data_first150k.csv'
file = pd.read_csv(data_name)
data = pd.DataFrame(file)

data = data.dropna()
data = data.drop(['Unnamed: 0', 'description', 'designation', 'region_1', 'region_2', 'winery'], axis=1)
data_US = data.drop(index=(data.loc[(data['country']!='US')].index))


# 接下来，我们统计省份属性的词频，代码及结果如下：

# In[209]:


counts = data_US['province'].value_counts()
print(counts)


# 可以看出，California，Washington，Oregon，New York出现次数远远大于其他省份的出现次数，因此我们将除却这四个城市的其他城市替换为others。代码如下，

# In[210]:


data_US.loc[(data_US['province']!= 'California') & (data_US['province']!= 'Washington') & (data_US['province']!= 'Oregon') & (data_US['province']!= 'New York'), 'province'] = 'Other province'


# In[211]:


counts = data_US['province'].value_counts()
print(counts)


# 按照上述依次对variety属性进行处理，代码如下：

# In[212]:


counts = data_US['variety'].value_counts()
print(counts)


# In[213]:


data_US.loc[(data_US['variety']!= 'Pinot Noir') & (data_US['variety']!= 'Cabernet Sauvignon') & (data_US['variety']!= 'Chardonnay') & (data_US['variety']!= 'Syrah') & (data_US['variety']!= 'Zinfandel'), 'variety'] = 'Other variety'


# In[214]:


counts = data_US['variety'].value_counts()
print(counts)


# 下面，我们对数值属性进行处理。在这里，我们使用四分数，中位数，以及75%分位数将数据划分为1，2，3，4类。代码如下，

# 由于points与price是数值属性，我们首先计算US的points与price的五数概括，代码及结果如下：

# In[208]:


data_US.describe()


# 注意：在这里我们将小于price的四分数的price值分箱为1，将介于四分位数与中位数之间的price值分箱为2，将介于中位数与四分之三位数之间的price值分箱为3，将大于四分之三分数的值分箱为4.代码如下：

# In[215]:



data_US.loc[data_US['price']<19,'price']=1
data_US.loc[(data_US['price']>=19) & (data_US['price']<28), 'price' ]=2
data_US.loc[(data_US['price']>=28) & (data_US['price']<41), 'price' ]=3
data_US.loc[(data_US['price']>=41), 'price'] = 4


# In[216]:


counts = data_US['price'].value_counts()
print(counts)


# 注意：在这里我们将小于points的四分数的price值分箱为10，将介于四分位数与中位数之间的points值分箱为20，将介于中位数与四分之三位数之间的points值分箱为30，将大于四分之三分数的points值分箱为40.代码如下：

# In[217]:


data_US.loc[data_US['points']<85,'points']=10
data_US.loc[(data_US['points']>=85) & (data_US['points']<88), 'points' ]=20
data_US.loc[(data_US['points']>=88) & (data_US['points']<90), 'points' ]=30
data_US.loc[(data_US['points']>=90), 'points'] = 40


# In[218]:


counts = data_US['points'].value_counts()
print(counts)


# In[219]:


data_US_copy = data_US.copy(deep = False)


# In[220]:


data_US = data_US.drop(['country'], axis=1)


# In[221]:


data_US_copy = data_US_copy.drop(['country'], axis=1)


# In[222]:


data_US_copy['price'] = data_US_copy['price'].apply(str)
data_US_copy['points'] = data_US_copy['points'].apply(str)


# 我们将处理之后的表格数据转换为字符串类型，代码如下：

# In[223]:


all_transactions = np.array(data_US_copy)


# In[224]:


all_transactions = all_transactions.tolist()


# 我们将转换为字符串类型的数据转换为可以进行频繁模式挖掘的数据类型，代码如下：

# In[225]:


from prettytable import PrettyTable
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

trans_encoder = TransactionEncoder() # Instanciate the encoder
trans_encoder_matrix = trans_encoder.fit(all_transactions).transform(all_transactions)
trans_encoder_matrix = pd.DataFrame(trans_encoder_matrix, columns=trans_encoder.columns_)


# In[226]:


trans_encoder_matrix.head()


# ## 使用FP growth方法对数据集进行挖掘

# In[254]:


import time
def perform_rule_calculation(transact_items_matrix, rule_type="fpgrowth", min_support=0.001):
    """
    desc: this function performs the association rule calculation 
    @params:
        - transact_items_matrix: the transaction X Items matrix
        - rule_type: 
                    - apriori or Growth algorithms (default="fpgrowth")
                    
        - min_support: minimum support threshold value (default = 0.001)
        
    @returns:
        - the matrix containing 3 columns:
            - support: support values for each combination of items
            - itemsets: the combination of items
            - number_of_items: the number of items in each combination of items
            
        - the excution time for the corresponding algorithm
        
    """
    start_time = 0
    total_execution = 0
    
    if(not rule_type=="fpgrowth"):
        start_time = time.time()
        rule_items = apriori(transact_items_matrix, 
                       min_support=min_support, 
                       use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Apriori!")
        
    else:
        start_time = time.time()
        rule_items = fpgrowth(transact_items_matrix, 
                       min_support=min_support, 
                       use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Fp Growth!")
    
    rule_items['number_of_items'] = rule_items['itemsets'].apply(lambda x: len(x))
    
    return rule_items, total_execution


def compute_association_rule(rule_matrix, metric="lift", min_thresh=1):
    """
    @desc: Compute the final association rule
    @params:
        - rule_matrix: the corresponding algorithms matrix
        - metric: the metric to be used (default is lift)
        - min_thresh: the minimum threshold (default is 1)
        
    @returns:
        - rules: all the information for each transaction satisfying the given metric & threshold
    """
    rules = association_rules(rule_matrix, 
                              metric=metric, 
                              min_threshold=min_thresh)
    
    return rules


# In[232]:


def plot_metrics_relationship(rule_matrix, col1, col2):
    """
    desc: shows the relationship between the two input columns 
    @params:
        - rule_matrix: the matrix containing the result of a rule (apriori or Fp Growth)
        - col1: first column
        - col2: second column
    """
    fit = np.polyfit(rule_matrix[col1], rule_matrix[col2], 1)
    fit_funt = np.poly1d(fit)
    plt.plot(rule_matrix[col1], rule_matrix[col2], 'yo', rule_matrix[col1], 
    fit_funt(rule_matrix[col1]))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} vs {}'.format(col1, col2))


# In[233]:


def compare_time_exec(algo1=list, alg2=list):
    """
    @desc: shows the execution time between two algorithms
    @params:
        - algo1: list containing the description of first algorithm, where
            
        - algo2: list containing the description of second algorithm, where
    """
    
    execution_times = [algo1[1], algo2[1]]
    algo_names = (algo1[0], algo2[0])
    y=np.arange(len(algo_names))
    
    plt.bar(y,execution_times,color=['orange', 'blue'])
    plt.xticks(y,algo_names)
    plt.xlabel('Algorithms')
    plt.ylabel('Time')
    plt.title("Execution Time (seconds) Comparison")
    plt.show()


# ### 计算数据集的频繁模式

# In[255]:


fpgrowth_matrix, fp_growth_exec_time = perform_rule_calculation(trans_encoder_matrix) # Run the algorithm
print("Fp Growth execution took: {} seconds".format(fp_growth_exec_time))


# In[256]:


fpgrowth_matrix.head()


# In[259]:


fpgrowth_matrix_choose = fpgrowth_matrix[(fpgrowth_matrix['number_of_items']>=2) &(fpgrowth_matrix['support']>=0.1) ]
print(fpgrowth_matrix_choose)


# In[236]:


fpgrowth_matrix.tail()


# ### 计算关联规则及评价指标

# In[246]:


fp_growth_rule_lift = compute_association_rule(fpgrowth_matrix)


# #### lift指标
# 下面，我们将对指标lift进行分析，首先将得到的规则按照conviction进行排序，代码如下，

# In[272]:


fp_growth_rule_lift_sort = fp_growth_rule_lift.sort_values('lift',axis = 0,ascending = False)


# In[280]:


fp_growth_rule_lift_sort.head(20)


# 通过将lift分数从小到大进行排序，我们可以得到看到上面显示的20条关联规则的lift值很高，代表我们可能找到了有用的关联规则。下面对这个20条关联规则进行分析。通过上述表格，我们可以清楚的看到，葡萄 Pinot Noir与省份Oregon存在非常强的关联关系。而 Pinot Noir（黑皮诺）恰恰只生长在美国的Oregon（俄亥冈州）。除此之外，我们还可以看到对于由Pinot Noir生产出来的葡萄酒价格与分数有着较强的关联，即价格越高分数越高，反之亦然。下面我们将绘制lift与置信度之间关系的曲线，代码如下：

# In[247]:


plot_metrics_relationship(fp_growth_rule_lift, col1='lift', col2='confidence')


# #### conviction指标
# 下面，我们将对指标conviction进行分析，首先将得到的规则按照conviction进行排序，代码如下，

# In[281]:


fp_growth_rule_conviction_sort = fp_growth_rule_lift.sort_values('conviction',axis = 0,ascending = False)
fp_growth_rule_conviction_sort.head(20)


# 通过上述表格，我们可以看到，Zinfandel与California存在着较强的相关关系，并且，可以注意到Zinfandel是自变量，California是因变量。其中原因为Zinfandel在美国地区均是由California地区所生产的。下面我们将分析conviction指标与置信度之间的关系，代码如下，

# In[293]:


def plot_metrics_relationship_1(rule_matrix, col1, col2):
    """
    desc: shows the relationship between the two input columns 
    @params:
        - rule_matrix: the matrix containing the result of a rule (apriori or Fp Growth)
        - col1: first column
        - col2: second column
    """
#     fit = np.polyfit(rule_matrix[col1], rule_matrix[col2], 1)
#     fit_funt = np.poly1d(fit)
    plt.plot(rule_matrix[col1], rule_matrix[col2], 'yo' )
#     fit_funt(rule_matrix[col1]))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} vs {}'.format(col1, col2))
           
plot_metrics_relationship_1(fp_growth_rule_lift, col1='conviction', col2='confidence')


# 我们可以看到，指标conviction与confidence存在较强的相关关系，即conviction越大，confidence越大。

# #### leverage指标
# 下面，我们将对指标leverage进行分析，首先将得到的规则按照leverage进行排序，代码如下，

# In[283]:


fp_growth_rule_conviction_sort = fp_growth_rule_lift.sort_values('leverage',axis = 0,ascending = False)
fp_growth_rule_conviction_sort.head(20)


# 我们可以看到，与上述两个指标不同。通过对leverage指标进行分析，可以得到以下的结论：
# * 价格与分数存在着较强的关联关系，并且，当价格很高时，得分通常也比较高
# * 由Pinot Noir制成的葡萄酒价格与得分往往很高，其主要原因为Pinot Noir同样也被看作是天性娇弱的贵族葡萄品种。它早熟、皮薄、色素低、产量少, 适合较寒冷地区, 然而发芽早的特点又使其对霜冻极为敏感, 皮薄使它对强光的耐受性差, 也使它对病菌及潮湿的抵抗力差, 易发生腐烂, 造成大量减产
# 

# 下面我们将分析conviction指标与置信度之间的关系，代码如下，

# In[294]:


plot_metrics_relationship(fp_growth_rule_lift, col1='leverage', col2='confidence')


# 下面，我们将对置信度进行分析，代码如下：

# In[295]:


fp_growth_rule = compute_association_rule(fpgrowth_matrix, metric="confidence", min_thresh=0.2)
fp_growth_rule.head()


# 通过上述表格，我们可以看到California与最高的酒的得分数存在关联规则。

# ### 分析
# 通过使用FP growth方法对关联规则进行挖掘，我们可以得到下面的结论：
# * 葡萄 Pinot Noir与省份Oregon存在非常强的关联关系。而 Pinot Noir（黑皮诺）恰恰只生长在美国的Oregon（俄亥冈州）。
# * 由Pinot Noir生产出来的葡萄酒价格与分数有着较强的关联，即价格越高分数越高，反之亦然。
# * Zinfandel与California存在着较强的相关关系，并且，可以注意到Zinfandel是自变量，California是因变量。其中原因为Zinfandel在美国地区均是由California地区所生产的。
# * 价格与分数存在着较强的关联关系，并且，当价格很高时，得分通常也比较高
# * 由Pinot Noir制成的葡萄酒价格与得分往往很高，其主要原因为Pinot Noir同样也被看作是天性娇弱的贵族葡萄品种。它早熟、皮薄、色素低、产量少, 适合较寒冷地区, 然而发芽早的特点又使其对霜冻极为敏感, 皮薄使它对强光的耐受性差, 也使它对病菌及潮湿的抵抗力差, 易发生腐烂, 造成大量减产。
# * California与最高的酒的得分数存在关联规则。

# ## 使用Apriori 方法对数据集进行挖掘

# In[297]:


apriori_matrix, apriori_exec_time = perform_rule_calculation(trans_encoder_matrix, rule_type="apriori")


# ### 计算数据集的频繁模式

# In[298]:


apriori_matrix.head()


# In[299]:


apriori_matrix.tail()


# ### 计算关联规则及评价指标

# In[300]:


apriori_rule_lift = compute_association_rule(apriori_matrix)


# #### lift指标
# 下面，我们将对指标lift进行分析，首先将得到的规则按照lift进行排序，代码如下，

# In[301]:


apriori_rule_lift_sort = apriori_rule_lift.sort_values('lift',axis = 0,ascending = False)


# In[302]:


apriori_rule_lift_sort.head(20)


# 在这里，我们可以得到与FP growth相似的结论。下面，我们将绘制lift与置信度之间关系的图像，代码如下：

# In[308]:


plot_metrics_relationship(apriori_rule_lift, col1='lift', col2='confidence')


# #### conviction指标
# 下面，我们将对指标lift进行分析，首先将得到的规则按照conviction进行排序，代码如下，

# In[307]:


apriori_rule_lift_sort = apriori_rule_lift.sort_values('conviction',axis = 0,ascending = False)


# In[306]:


apriori_rule_lift_sort.head(20)


# 在这里，我们可以得到与FP growth相似的结论。下面，我们将绘制conviction与置信度之间关系的图像，代码如下：

# In[310]:



plot_metrics_relationship_1(apriori_rule_lift, col1='conviction', col2='confidence')


# #### leverage指标
# 下面，我们将对指标lift进行分析，首先将得到的规则按照leverage进行排序，代码如下，

# In[311]:


apriori_rule_lift_sort = apriori_rule_lift.sort_values('leverage',axis = 0,ascending = False)


# In[312]:


apriori_rule_lift_sort.head(20)


# 在这里，我们可以得到与FP growth相似的结论。下面，我们将绘制leverage与置信度之间关系的图像，代码如下：

# In[313]:


plot_metrics_relationship(apriori_rule_lift, col1='leverage', col2='confidence')


# ### 分析
# 通过使用Apriori 方法对关联规则进行挖掘，我们可以得到下面的结论：
# * 葡萄 Pinot Noir与省份Oregon存在非常强的关联关系。而 Pinot Noir（黑皮诺）恰恰只生长在美国的Oregon（俄亥冈州）。
# * 由Pinot Noir生产出来的葡萄酒价格与分数有着较强的关联，即价格越高分数越高，反之亦然。
# * Zinfandel与California存在着较强的相关关系，并且，可以注意到Zinfandel是自变量，California是因变量。其中原因为Zinfandel在美国地区均是由California地区所生产的。
# * 价格与分数存在着较强的关联关系，并且，当价格很高时，得分通常也比较高
# * 由Pinot Noir制成的葡萄酒价格与得分往往很高，其主要原因为Pinot Noir同样也被看作是天性娇弱的贵族葡萄品种。它早熟、皮薄、色素低、产量少, 适合较寒冷地区, 然而发芽早的特点又使其对霜冻极为敏感, 皮薄使它对强光的耐受性差, 也使它对病菌及潮湿的抵抗力差, 易发生腐烂, 造成大量减产。
# * California与最高的酒的得分数存在关联规则。

# In[ ]:




