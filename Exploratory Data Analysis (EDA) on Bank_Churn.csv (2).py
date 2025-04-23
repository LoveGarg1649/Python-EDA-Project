#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('Bank_Churn.csv')


# In[4]:


data


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe(include = 'all')


# In[9]:


data.isnull().sum()


# In[10]:


data.Geography.nunique()


# In[11]:


data.Geography.unique()


# In[12]:


# Counting Number of active members
activeMembers = data[data['IsActiveMember'] == 1].shape[0]
print(activeMembers)


# Since, number of active members are equal to 5151. Hence the ratio of activeMembers/non-activeMembers is 0.5151.

# In[13]:


maleCount = data[data['Gender'] == 'Male'].shape[0]
print(maleCount)


# In[14]:


active_members = data[data["IsActiveMember"] == 1]
gender_counts = active_members["Gender"].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
for i, value in enumerate(gender_counts.values):
    plt.text(i, value + 50, str(value), ha='center', fontweight='bold')
plt.title("Number of Active Members by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Active Members")
plt.tight_layout()
plt.show()


# Since, total active members are 5151. In 5151, 2867 belongs to male gender category and the rest 2284 belongs to female.

# In[16]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='CreditScore', data=data)
plt.title('Credit Score Distribution by Churn Status')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='CreditScore', hue='Exited', kde=True, element='step')
plt.title('Credit Score Distribution')
plt.show()


# Boxplot Insights (Left Plot)<br>
# i.) The median credit scores for both churned (Exited = 1) and non-churned (Exited = 0) customers are quite similar.<br>
# ii.) Both groups have a similar interquartile range (middle 50% of data), though the churned group has slightly more lower-end outliers (credit scores below ~400).<br>
# iii.) Overall distribution spread is comparable, suggesting that credit score alone may not strongly distinguish churn behavior.
# <br>
# 
# Histogram & KDE Insights (Right Plot)<br>
# i.) The non-churned customers (Exited = 0) show a higher count across most credit score ranges, especially around the 650–700 mark.<br>
# ii.) The churned group (Exited = 1) has a more flattened distribution, indicating fewer customers with high credit scores.<br>
# iii.) There is a slight tendency for churned customers to have slightly lower credit scores on average, but the overlap is significant.

# In[38]:


plt.figure(figsize=(10, 6))
sns.jointplot(x='Age', y='CreditScore', data=data, hue='Exited', kind='kde', palette={0: 'blue', 1: 'red'})
plt.suptitle('CreditScore vs. Age by Churn Status', y=1.02)
plt.show()


# Conclusion:<br>
# i.) CreditScore Distribution:<br>
# a.) Most customers (both retained and churned) have credit scores between 600-800<br>
# b.) The density appears slightly higher in the 650-750 range<br>
# ii.) Age Distribution:<br>
# a.) Retained customers (blue) show a relatively even age distribution<br>
# b.) Churned customers (red) appear more concentrated in older age groups (likely 40+)<br>
# iii.) Relationship Patterns:<br>
# a.) Older customers with mid-range credit scores (600-700) show higher churn density<br>
# b.) Younger customers across all credit scores are more likely to be retained<br>
# c.) Very high credit scores (>800) show lower churn regardless of age<br>
# 
# Recommendation:<br>
# Targeted Marketing: Middle-aged customers (30–50) with moderate credit scores (600–750) form the largest segment—ideal for retention strategies.

# In[34]:


avg_credit_by_geo = df.groupby('Geography')['CreditScore'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Geography', y='CreditScore', data=avg_credit_by_geo, palette='viridis')
plt.title('Average Credit Score by Geography', fontsize=16)
plt.xlabel('Geography', fontsize=12)
plt.ylabel('Average Credit Score', fontsize=12)
plt.ylim(600, 700)
for index, row in avg_credit_by_geo.iterrows():
    plt.text(index, row['CreditScore'], f"{row['CreditScore']:.1f}", 
             ha='center', va='bottom', fontsize=12)
plt.show()


# Conclusion:<br>
# i.) Germany has the highest average credit score (≈ 650), suggesting customers in Germany tend to have better creditworthiness compared to other regions.<br>
# ii.) France follows closely (≈ 640), indicating relatively good credit profiles among French customers.<br>
# iii.) Spain has the lowest average credit score (≈ 630), which may imply higher credit risk or different lending standards in this region.

# In[35]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='Tenure', data=data)
plt.title('Tenure Distribution by Status')
plt.subplot(1, 2, 2)
sns.countplot(x='Tenure', hue='Exited', data=data)
plt.title('Tenure Distribution')
plt.show()


# Conclusion:<br>
# i.) Tenure is not a strong predictor of churn on its own.<br>
# ii.) Customers churn at all stages of their relationship with the bank.<br>
# iii.) Might be more valuable when combined with other features (e.g., products, activity status).

# In[36]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Exited', y='EstimatedSalary', data=data)
plt.title('Salary Distribution by Status')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='EstimatedSalary', hue='Exited', kde=True, element='step')
plt.title('Salary Distribution')
plt.show()


# ✅ Conclusion:<br>
# i.) Estimated Salary is not a strong predictor of churn.<br>
# ii.) Churn is independent of salary level — customers across low, medium, and high salary brackets behave similarly.<br>
# iii.) This feature is likely not important for churn prediction on its own.

# In[37]:


projects_by_gender = df.groupby('Gender')['NumOfProducts'].sum().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='NumOfProducts', data=projects_by_gender, palette='Set2')
plt.title('Total Number of Products by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Total Number of Products', fontsize=12)
for index, row in projects_by_gender.iterrows():
    plt.text(index, row['NumOfProducts'], int(row['NumOfProducts']), 
             ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Geography', y='Exited', hue='Gender', data=data)
plt.title('Churn Rate by Geography and Gender')
plt.show()


# In[23]:


plt.figure(figsize=(10, 8))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# Conclusion:<br>
# i.) Churn is more behavioral (activity, product use) than financial (salary, credit score).<br>
# ii.) Age is the most influential positive correlate of churn.<br>
# Being active and engaged with multiple products can reduce churn.<br>

# Recommendations for the Bank:<br>
# i.) Targeted Retention Programs: Focus on older customers and customers of age between 30 to 40, especially females in Germany<br>
# 
# ii.) Product Bundling: Encourage customers to have 2-3 products (where churn is lowest)<br>
# 
# iii.) Engagement Strategies: Increase engagement with inactive members to reduce churn<br>
# 
# iv.) High-Balance Customers: Develop special programs for customers with high balances who may be more sensitive to service quality<br>
