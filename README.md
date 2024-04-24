Customer Segmentation and Consumer Behavior Analysis
Description
This project aims to analyze customer behavior and segment customers based on RFM (Recency, Frequency, Monetary) using real sales data from a UK-based retailer. The dataset contains transactions occurring between 01/12/2010 and 09/12/2011. The company primarily sells unique all-occasion gifts, and many customers are wholesalers.

Data Source
The dataset used in this project is sourced from the UCI Machine Learning Repository.

Approach
Data Cleaning:
Convert attributes to the proper data type:
Convert 'Monetary', 'Frequency', and 'Recency' attributes to appropriate numerical data types.
Convert 'InvoiceDate' to datetime datatype for proper time-based analysis.
Scaling: Scale the data as necessary for model building.
Model Building: Utilize the K-means clustering algorithm to segment customers based on RFM.
Elbow Method: Determine the optimal number of clusters using the elbow method.
Result Analysis: Analyze the results of the clustering algorithm to understand customer segments.
Model Performance Check:
The k-means clustering algorithm being an unsupervised learning algorithm, we can perform a quick visual check on the model's performance based on the visualization chart.
Business Strategy
Cluster Analysis:
Cluster 1 may already be dominated by others. Focus on increasing sales in the 2 other clusters (0 & 2) through suitable competitive positioning, pricing strategies, cohesive sales & marketing efforts, promotions, bundling, etc.
Cluster 1 is still important. Aim to increase sales through suitable marketing efforts, promotions, bundling, etc.
Opportunities for Growth:
Customer segments 0 & 2 have opportunities for growth and future expansion.
Insights
Buying Analysis: The UK alone contributes to 82% of total revenue, indicating the importance of the UK market for the retailer.
Geographic Expansion: Consider expanding to other countries with significant sales potential, such as Netherlands, EIRE, Germany, and France, for future geographic expansion.
Contributing
Contributions to improve the analysis techniques, expand insights, or optimize the code are welcome. 
