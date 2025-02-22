#!/usr/bin/env python
# coding: utf-8

# ###### Customer segmentation and consumer behavior

# #### This is a real sales data set of a UK based retailer , all the transaction occurring between 01/12/2010 and 09/12/2011 for a UK - bases , the company mainly sells unique all - occasion gists . many customers of the company are wholeselers.

# #### Data source : https://archive.ics.uci.edu/dataset/352/online+retail

# ### Segment  the customers based on RFM so that the company can target its customers effciently 

# ### R ( Recency ) - Number of days since last purchase 

# ### F (Frequency ) - Number of trancsactions 

# ### M (Monetary) - Total amount of transactions ( revenue contributed)

# In[4]:


# Required libraries for clustering 
import numpy as np 
import pandas  as  pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt 
import sklearn 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')


# In[5]:


online_retail = pd.read_excel("D:\yash\Intern\Online Retail.xlsx")


# In[6]:


online_retail


# In[7]:


#shape online_retail
online_retail.shape


# In[8]:


#info 
online_retail.info()


# # Data cleaning 

# In[9]:


# calculating the  missing values % contribution in datarame


# In[10]:


null = round(100*(online_retail.isnull().sum())/len(online_retail),2)


# In[11]:


null


# In[12]:


online_retail = online_retail.dropna()


# In[13]:


online_retail.shape


# In[14]:


#changing the datatype od customer id 
online_retail['CustomerID'] = online_retail['CustomerID'].astype(str)


# #### New Attribute : Monetary 

# In[15]:


online_retail['Amount'] = online_retail['Quantity']*online_retail['UnitPrice']
monetary = online_retail.groupby('CustomerID')['Amount'].sum()
monetary = monetary.reset_index()
monetary


# #### New Attributes : Frequency 

# In[16]:


frequency = online_retail.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID' , 'frequency']
frequency.head()


# In[17]:


# Merging the two Attibutes 


# In[18]:


rfm = pd.merge(monetary,frequency , on='CustomerID' , how = 'inner') # base on CustomerID , merge - inner merge 


# In[19]:


rfm


# #### New Attributes : Recency 
# #### Convert to datetime to proper datatype 

# In[20]:


online_retail['InvoiceDate'] = pd.to_datetime(online_retail['InvoiceDate'],format='%d-%m-%y %H:%M')


# In[21]:


online_retail['InvoiceDate']


# In[22]:


#compute the maximum date  to know the last transaction date 
max_date = max(online_retail['InvoiceDate'])
max_date


# In[23]:


#Compute the difference between max date and  transcsaction date 
online_retail['dif'] = max_date - online_retail['InvoiceDate']
online_retail


# In[24]:


recency= online_retail.groupby('CustomerID')['dif'].min()
recency = recency.reset_index()
recency.head()


# In[25]:


#Extract number of days only 
recency['dif'] = recency['dif'].dt.days 
recency.head()


# In[26]:


rfm = pd.merge(rfm,recency, on =
               'CustomerID' , how = 'inner')
rfm.columns = ['CustomerID','Amount','frequency','Recency']
rfm


# In[27]:


#Outlier analysis of amount,frequency,recency 
attributes = ['Amount','frequency','Recency']
sns.boxplot(data = rfm[attributes])
plt.title('Outliers')
plt.ylabel('Range')
plt.xlabel('Attributes',fontweight = 'bold')


# In[28]:


#Outlier analysis of amount,frequency,recency 
attributes = ['Amount','frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes] , orient = "v",palette = "Set2" , whis = 1.5 , saturation = 1 , width = 0.7)
plt.title('Outliers',fontsize = 14 ,fontweight='bold')
plt.ylabel('Range',fontweight = 'bold')
plt.xlabel('Attributes',fontweight = 'bold')


# In[29]:


Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1 
rfm = rfm[(rfm.Amount >= Q1 - 1.5* IQR) & ( rfm.Amount <= Q3 + 1.5*IQR)]


Q1 = rfm.Recency.quantile(0.05)
Q3 =  rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5* IQR) & ( rfm.Recency <= Q3 + 1.5*IQR)]


Q1 = rfm.frequency.quantile(0.05)
Q3 = rfm.frequency.quantile(0.95)
IQR = Q3 - Q1 
rfm = rfm[(rfm.frequency >=Q1 - 1.5*IQR) & (rfm.frequency <= Q3 + 1.5* IQR)]


# ## Scaling 

# In[30]:


#our machine learing model d'not byus for high value like amount


# In[31]:


rfm_df = rfm[['Amount','Recency','frequency']]
scaler = StandardScaler()
#fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# ## Model Building

# ### Elbow method for  right numbers  of clusters 

# In[32]:


wssd = []
range_n_clusters = [2,3,4,5,6,7,8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters,max_iter = 50)
    kmeans.fit(rfm_df_scaled)
    
    wssd.append(kmeans.inertia_)
    
plt.plot(wssd)


# In[33]:


# In the  particular case use  2,3 number  of cluster 


# #### FInal model with k = 3 

# In[34]:


kmeans = KMeans(n_clusters=3 , max_iter = 300)
kmeans.fit(rfm_df_scaled)


# In[35]:


kmeans.labels_ #data labling (decide to belong data for cluster)


# In[36]:


kmeans.predict(rfm_df_scaled)


# In[37]:


#assign the label 
rfm['Cluster_Id'] = kmeans.predict(rfm_df_scaled)
rfm.head()


# In[38]:


rfm.head(100)


# In[39]:


#box plot cluster id vs amount 
sns.stripplot(x = 'Cluster_Id' , y = 'Amount' , data = rfm)



# number 1 cluster  people are investing more money


# In[40]:


sns.stripplot(x='Cluster_Id' , y ='frequency' ,  data = rfm)
# number 1 cluster  people are frequently buy


# In[41]:


sns.stripplot(x='Cluster_Id',y='Recency' , data = rfm)
# number 1 cluster  people low receny 


# The k-means clustering algorithm being a un-supervised learning algorithm, we can perform a quick visual check on the model's performance based on the visualization chart.

# #### Business Strategy:

# cluster 1 may already be dominated by other , can try to increaese sales in the 2 other cluster ( 0 & 2) through suitable competitive positioning, pricing stratgey, cohesive sales & marketing efforts, promotions, bundling etc.
# Also cluster 1 important - try to more increaese sales through suitable  marketing efforts, promotions, bundling 
#  

# Customer segment  0 & 2 have opportunities for growth and future expansion

# ## Buying Analysis

# In[42]:


online_retail


# In[43]:


# Removing missing data

online_retail = online_retail.dropna()
online_retail.info()


# In[44]:


# Quantity
online_retail.sort_values("Quantity", ascending = False).head(5)


# In[45]:


online_retail.sort_values("Quantity", ascending = False).tail(5)


# Quantity is negative may be due to discounts, damaged goods, thrown away etc. I shall remove these values.

# In[46]:


mask = online_retail["Quantity"] > 0

online_retail = online_retail [mask]
online_retail.sort_values("Quantity", ascending = False).tail(5)


# In[47]:


# For some customers, their information on country is unspecified, lets filter those out

mask = online_retail["Country"] != "Unspecified"
online_retail = online_retail[mask]


# In[48]:


# Creating new column - Revenue in $

online_retail["Revenue"] = online_retail["Quantity"]*online_retail["UnitPrice"]
online_retail.head(10).sort_values("Revenue", ascending = False).head(5)


# In[49]:


online_retail.sort_values("Revenue", ascending = False).tail(5)

# Revenue is 0 for some quantities, as they may have been given away as promotional offers, I shll remove these as well


# In[50]:


mask = online_retail["Revenue"] > 0

dataset = online_retail [mask]
online_retail.sort_values("Revenue", ascending = False).head(5)


# In[51]:


online_retail.sort_values("Revenue", ascending = False).tail(5)


# In[52]:


dataset["StockCode"] = dataset["StockCode"].astype('object')
dataset["CustomerID"] = dataset["CustomerID"].astype('object')

dataset.info()


# In[53]:


# Customer ID

print("Top 20% of customers driving 80% of revenue:")
online_retail["CustomerID"].nunique()

online_retail.groupby("CustomerID").agg({"Revenue": "sum"}).sort_values("Revenue", ascending = False).head(20).plot(kind = "bar")
                                
plt.show()


# In[54]:


# Customer IDs of top segments

dataset1 = online_retail[["CustomerID", "Revenue"]]

print("Top customer segment IDs")
dataset2 = dataset1.groupby("CustomerID").agg({"Revenue": "sum"}).sort_values("Revenue", ascending = False)

dataset2.head(5)


# In[55]:


h = dataset1["CustomerID"].nunique()
i = round(0.25*h)
print("# of customers in top 25% power segment:", i, "out of", h)

j = dataset1.groupby("CustomerID").agg({"Revenue": "sum"}).sort_values("Revenue", ascending = False).head(i).sum()
k = dataset1["Revenue"].sum()

l = j/k*100
print("Total sales resulting from the top product segment:", round(list(l)[0]), "%")


# 1084 out of total 3877, top 25% of customer segments result in 79% of total $ sales amount.

# While the top 25 % customers can be targetted for potential up sell and cross sell potential.
# 
# Considering that the retail industry is very competitive, the remaining 75% of customers can be targetted for future expansion by coordination with sales & marketing teams, discounts, pricing stratagy, bundling products, etc.

# In[56]:


# Segmenting sales by geographic location

dataset1 = dataset[["Country", "Revenue"]]

print("Countries:")
dataset1["Country"].nunique()
print("Top 10 countries by $ sales:")
dataset2 = dataset1.groupby("Country").agg({"Revenue": "sum"}).sort_values("Revenue", ascending = False)
dataset2.head(5).plot(kind = "bar")


# In[57]:


print("Bottom 10 countries by $ sales:")

dataset2.tail(10).plot(kind = "bar")

# Analysis


# In[58]:


# Identifying the top 20 % geographic locations driving 80 % of $ sales

h = dataset1["Country"].nunique()
i = 1

print("# of products in UK:")

j = dataset2.head(i).sum()
k = dataset1["Revenue"].sum()

l = j/k*100
print("Total $ sales resulting from UK:", round(list(l)[0]), "%")


# In[59]:


dataset2.head(5)


# UK alone results in 82% of total $ revenue which is expected for a UK based retailer.

# The senior management for the retail company should consider expanding to the other countries (fter UK) where they have significant sales such as Neatherland, EIRE, Germany & France for future geographic expansion.

# In[ ]:
..



