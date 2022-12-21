# importing libraries
import datetime as dt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# display setting
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
###################
# DATA PREPARATION
###################
"""
Information about Dataset:
 This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.
 you can access more information about dataset and download from:
  https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
  
Attribute Information:
    - Invoice: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
    - StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
    - Description: Product (item) name. Nominal
    - Quantity: The quantities of each product (item) per transaction. Numeric.
    - InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated
    - UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
    - CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
    - Country: Country name. Nominal. The name of the country where a customer resides.

"""

# reading dataset and creating a copy of it
df_ = pd.read_excel("datasets/online_retail_II.xlsx")
df = df_.copy()
df.head()
df.shape

# number of unique product :
df["Description"].nunique()

# how many of each product were sold
df["Description"].value_counts().head()

# What is the most ordered product?
df.groupby("Description").agg({"Quantity": "sum"}).head()

# missing value check
df.isnull().sum

# Customers with missing Customer ID cannot be included in the segmentation process, so these customers must be removed from the dataset.
df.dropna(inplace =True)

# adding the total price on the basis of products to the data set as a variable
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

# descriptive statistics of the dataset
df.describe().T
# as you can see there ar negative values, because of the returns. we need to get rid of them.

# in Ivoice attirubute values started with "C" specify returned products.
df = df[~df["Invoice"].str.contains("C", na=False)]

df.describe().T

##########################
# CALCULATION RFM METRICS
##########################
# the dataset is former so we are going to set an analysis date:

# last date int the InvoiceDate
df["InvoiceDate"].max()

# 2 days after the last date
analysis_date = dt.datetime(2010, 12, 11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda invoice_date: (analysis_date - invoice_date.max()).days,
                                     "Invoice": lambda invoice: invoice.nunique(),
                                     "TotalPrice": lambda total_price: total_price.sum()})
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

# it is an unwanted situation that monetary to be 0, so they shold be eliminated.
rfm = rfm[rfm["monetary"] > 0]
rfm.describe().T

# RECENCY SCORE
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# FREQUENCY SCORE
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# MONETARY SCORE
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# RF SCORE
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.head()

# converting RF_SCORE to segments with Regex
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

# statistical informations of segments
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# How many segments are there and what percentage of them are.
segments_counts = rfm['segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='lightcoral')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['Can\'t loose']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )

plt.show(block=True)
