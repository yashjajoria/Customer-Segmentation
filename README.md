
# Customer Segmentation and Consumer Behavior Analysis

Description
This project aims to analyze customer behavior and segment customers based on RFM (Recency, Frequency, Monetary) using real sales data from a UK-based retailer. The dataset contains transactions occurring between 01/12/2010 and 09/12/2011. The company primarily sells unique all-occasion gifts, and many customers are wholesalers.


## Data Source
The dataset used in this project is sourced from the UCI Machine Learning Repository.(https://archive.ics.uci.edu/dataset/352/online+retail)


## Approach
1. Data Cleaning:
Convert attributes to the proper data type:
Convert 'Monetary', 'Frequency', and 'Recency' attributes to appropriate numerical data types.
Convert 'InvoiceDate' to datetime datatype for proper time-based analysis.

2. Scaling: 
 Scale the data as necessary for model building.

3. Model Building: 
Utilize the K-means clustering algorithm to segment customers based on RFM.

4. Elbow Method: 
Determine the optimal number of clusters using the elbow method.

5. Result Analysis: 
Analyze the results of the clustering algorithm to understand customer segments.

6. Business Strategy:
*Identify opportunities for increasing sales in specific customer clusters.

* Develop competitive positioning, pricing strategies, sales and marketing efforts, promotions, and bundling strategies tailored to each cluster.

* Consider customer segments with growth potential for future expansion.
## Insights
* Buying Analysis: 
The UK alone contributes to 82% of total revenue, indicating the importance of the UK market for the retailer.

* Geographic Expansion:
 Consider expanding to other countries with significant sales potential, such as Netherlands, EIRE, Germany, and France, for future geographic expansion.
## Business Strategy
* Cluster 1 may already be dominated by others. Increase sales in clusters 0 & 2 through suitable competitive positioning, pricing strategy, cohesive sales & marketing efforts, promotions, bundling, etc. Additionally, focus on increasing sales in cluster 1 through suitable marketing efforts, promotions, and bundling.
* Customer segments 0 & 2 have opportunities for growth and future expansion.
## Contributing
* Contributions to improve the analysis techniques, expand insights, or optimize the code are welcome. 
