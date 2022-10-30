import datetime as dt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)

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

# Reading Data
df_original = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = ["Year 2009-2010", "Year 2010-2011"])

# make a copy of the dataset
df1 = df_original["Year 2009-2010"]
df2 = df_original["Year 2010-2011"]
df = df1.append(df2)

df.head()
df.shape

# checkin missing values
df.isnull().sum()

# eliminating missing values
df.dropna(inplace=True)

#descriptive statistics of the dataset
df.describe().T
# as you can see there are negative values, because of the returned products. we need to get rid of them.

# in Ivoice attirubute values started with "C" specify returned products.
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# creating and adding the Total Price attribute to thr dataset
df["TotalPrice"] = df["Quantity"] * df["Price"]

##########
# CLTV Calculation
##########
cltv_c = df.groupby('Customer ID').agg({"Invoice": lambda x: x.nunique(),
                                     "Quantity": lambda x: x.sum(),
                                     "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

# profit margin:
cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

# Churn Rate:
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

# Purchase Frequency:
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

# Average Order Value:
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

# Customer Value:
cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

# CLTV (Customer Lifetime Value):
cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# sorting dataset based on from highest to lowest
cltv_c.sort_values(by = "cltv", ascending=False).head()



# Eliminating outliers with iqr
def outlier_threshold(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# define the analysis date
# since this is a formar dataset analysis data will be 2 days after the last date in the dataset
df["InvoiceDate"].max()
analysis_date = dt.datetime(2011, 12, 11)


#########
# Preparation of Lifetime Data Structure
#########
# recency, tenure, frequency and monetary_value are the values are customer specific values:
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days / 7,
                                                         lambda InvoiceDate: (analysis_date - InvoiceDate.min()).days / 7],
                                         "Invoice": lambda Invoice: Invoice.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

# we are calculating recency and tenure values based on InvoiceDate, and we want them in the week format. So i divide them with 7.

# drop the level0 varible naming
cltv_df.columns = cltv_df.columns.droplevel(0)

# renaming attributes
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

# i mentioned that recency is the average profit per transaction so we need to divide the monetary value we calculated above to frequency:
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# also it is mentioned that frequency should be greater than 1:
cltv_df = cltv_df[cltv_df["frequency"] > 1]

cltv_df.head()

cltv_df.describe().T


###########
# BG/NBD Model
###########
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# the first 10 customer to we expect highest purchase:
bgf.conditional_expected_number_of_purchases_up_to_time(1, # number of weeks
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

# we can do same thing with predict:
bgf.predict(1, #number of weeks
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

# first 20 customer that we expect highest purchases for 1 mont:
bgf.predict(4, # 4 weeks = 1 month
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(20)

# expected purchases for 1 months
cltv_df["expected_purchase_1_month"] = bgf.predict(4,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

# expected purchases for 3 months
cltv_df["expected_purchase_3_month"] = bgf.predict(4 * 3,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

# expected purchases for 1 year
cltv_df["expected_purchase_1_year"] = bgf.predict(4 * 12,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

cltv_df.head()

# evaluation of predicted results
plot_period_transactions(bgf)
plt.show(block=True)

from lifetimes.plotting import plot_frequency_recency_matrix
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plot_frequency_recency_matrix(bgf)
plt.show(block=True)

#############
# Gamma-Gamma Submodel
#############

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##########
# Calculation of CLTV with BG/NBD an GG Model
##########
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,  # 3 month
                                   freq="W",  # frequency of T
                                   discount_rate=0.01)

#Customer ID exists as an index, to convert it to a varibale:
cltv = cltv.reset_index()

# merge cltv_df and cltv based on Customer ID
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# It is seen that the customer_life_time method makes the naming of the data it returns as clv, we will sort data from largest to smallest accordingly according to clv.
cltv_final.sort_values(by="clv", ascending=False).head(10)

#########
# Segmentation according to CLTV
#########
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.groupby("segment").agg({"count", "mean", "sum"})

_df_plt = cltv_final[["expected_average_profit", "segment"]]
plot_df = _df_plt.groupby("segment").agg("mean")
plot_df.plot(kind="bar", color="tomato",figsize=(15, 8))
plt.ylabel("mean")

plt.show(block=True)
