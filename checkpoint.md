# **DS 3010: Global Climate Health Project Checkpoint**

Group 3: Karlie Lynch, Nathan Bledsoe, Sean Skaritanov, Gianni Rosato

## **Summary & Descriptive Statistics**

We utilized the
[Global Climate-Health Impact Tracker (2015–2025) dataset](https://www.kaggle.com/datasets/sohumgokhale/global-climate-health-impact-tracker-2015-2025/data)
sourced from Kaggle. The dataset contains approximately 14,100 weekly
observations collected over an 11-year period from 25 countries, representing
eight global regions and three income levels. It includes 30 features capturing
a broad range of variables relevant to climate health analysis, designed to
support analyses of climate-related policy impacts.

After our cleaning process, the data reveals the following baseline conditions:

- The mean global temperature in the dataset is 8.6°C, with a wide variance
  (Min: \-20.7°C, Max: 38.3°C)
- The average PM2.5 concentration is 65.77 µg/m³, significantly higher than WHO
  guidelines, indicating widespread exposure to poor air quality
- By preserving outliers, we identified that while rare (mean \~0.15), extreme
  weather events such as heat waves and floods reach a maximum frequency of 5
  events per week in severely affected regions
- Respiratory disease rates average 70.0 cases per 10,000, showing a lot of
  variability (Std Dev: 15.2) which we aim to explain through climate factors

## **Data Cleaning Steps Taken**

All data cleaning was performed using Python (Pandas) through a local Python
script which we have stored on GitHub. Our process evolved to strictly preserve
environmental signals.

1. We converted the date column to datetime objects and extracted year, month,
   and week features for time-series analysis
2. We separated categorical identifiers (Country, Region, Income Level) to focus
   the statistical cleaning on numerical data
3. Initially, we applied a standard Interquartile Range (IQR) filter to all
   numeric columns. However, this erroneously removed relevant "extreme" climate
   events (e.g., heat waves, floods), resulting in zero-variance columns. For
   obvious reasons, these are important when doing climate-related data
   analysis. To fix this issue, we implemented a selective cleaning function; we
   applied IQR clipping only to continuous variables (e.g., GDP, Temperature,
   Disease Rates) to reduce noise, while strictly protecting event-flag columns
   (`heat_wave_days`, `extreme_weather_events`) to preserve the climate impact
   signals central to our hypothesis.

## **Exploratory Data Analysis & Insights**

Our preliminary analysis utilizing correlation matrices and scatterplots has
yielded three primary insights. _Note: these reference visualizations, which
we’ve included in the next section._

### **Air Quality is a Linear Predictor of Respiratory Illness**

As visualized in our scatterplot, there is a distinct positive linear
relationship between PM2.5 concentrations and respiratory disease rates. This
trend persists across income levels, though lower-middle-income nations (orange
in visual) often cluster in the higher-pollution/higher-disease quadrant.

### **Temperature Anomalies Strain Healthcare Systems**

Our correlation analysis reveals a moderate positive correlation (0.42) between
temperature\_celsius and heat\_related\_admissions. This suggests that as
baseline temperatures rise, the immediate burden on hospital admissions
increases, independent of other variables.

### **Economic Disparities in Climate Resilience**

A strong negative correlation (-0.62) exists between gdp\_per\_capita\_usdand
temperature\_celsius (likely due to geographic distribution of wealthy nations
in temperate zones), but notably, higher GDP is strongly correlated (0.83) with
the healthcare\_access\_index. This implies that wealthier nations are better
positioned to mitigate the health effects of the very climate anomalies they
face.

## **Sketch of Interaction in Final Data Product**

For our final deliverable, we are developing an interactive "Global
Climate-Health Dashboard" using Plotly Dash. The interface will feature:

1. A central choropleth map where users can visualize the
   `climate_vulnerability_index`. Clicking a country will filter the surrounding
   charts to that specific nation's data.
2. A control allowing users to switch views between "High Income," "Middle
   Income," and "Low Income" nations to see how the same climate event (e.g., a
   2°C temp rise) impacts health outcomes differently based on economic
   resilience.
3. A tool allowing users to offset the health data by 1–4 weeks. This allows for
   the exploration of "lag effects" (e.g., observing if cholera outbreaks spike
   2 weeks after a flood event rather than immediately).

## **Next Steps**

To complete the project by the final deadline, we will:

1. Generate lagged variables for health outcomes to statistically quantify the
   delay between an extreme weather event and the resulting public health crisis
2. Move beyond scatterplots to implement the geospatial visualizations required
   to show the regional clustering of vector-borne diseases
3. Migrate our static Seaborn charts into interactive Plotly components and
   deploy the layout locally for the final presentation
