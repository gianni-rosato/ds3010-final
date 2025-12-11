# Global Climate-Health Impact Tracker (2015-2025)

Group 6

Nathan Bledsoe, Gianni Rosato, Karlie Lynch, Sean Skaritanov

## What is the need? Who wants or benefits?

The global climate is becoming increasingly unpredictable and volatile. Vulnerable populations see immediate damage from unprecedented weather events and natural disasters, but long-term effects are starting to be seen as well. Personal health is affected by long-term climate variations. Additionally, indicators for droughts, flooding, and other variables are becoming increasingly crucial given the increased unpredictability in weather events. This culminates in a great need for a health impact tracking system that can monitor climate indicators to assess public health outcomes.

This system benefits everyone; the entire human population is susceptible to different environmental and personal risks. High-risk populations are most in need of this system to monitor personal health, to prepare for detrimental weather events, and adapt to changing weather patterns and conditions before it is too late to act. Many companies with a globalized structure are also good candidates to utilize this system. It would give more insight into the risks of going to different locations in the long term, as well as being more informed about their employees and customers in that region. Lastly, governments, policy makers, and NGOs have the most ability to use these predictive models to make actionable changes and impact the long-term infrastructure of mitigating the effects of climate change.

## What data (or datasets)?

We are using the [Global Climate-Health Impact Tracker](https://www.kaggle.com/datasets/sohumgokhale/global-climate-health-impact-tracker-2015-2025) (2015-2025) from Kaggle. This data set contains over 14,100 records collected over the span of 11 years. Data was collected weekly from 25 countries across 8 different regions and 3 different income levels. The 30 different features contain key information on climate indicators like temperature and extreme weather events. Socioeconomic factors including GDP and healthcare access, as well as personal health outcomes such as respiratory and cardiovascular health, and wellbeing factors like mental health and food security. This dataset was designed for climate-health research or to analyze the effect of different climate health polices across the globe.

## What is your "data science" toolkit? You should list specific tools / packages you will use.

We will be using Python to perform all data analysis tasks, coupled with Numpy, Pandas, Scipy, scikit-learn, Folium for visualizations, uv for package management, and Bash for additional scripting. Folium will be our maps provider, which allows us to easily produce web-ready leaflet maps to pair with our visualizations.

## Preliminary sketch of what you hope to build.

We are planning to build an interactive climate health tool to explore different climate and socioeconomic risk factors throughout the world with the use of an interactive world map. We will use Python, Folium, and other data analysis and visualization tools. Users will be able to explore how climate variables relate to health outcomes across the world from 2015-2025. We will generate a world map where each country in the dataset is represented. Users should be able to click on a country and see the predicted health risk based on climate factors. It would also be capable of displaying in-depth data such as predicted disease rate, climate statistics, and socioeconomic factors like GDP. We are in the process of refining our design and eventual implementation, but the idea is to produce a map-based visualization interface capable of easily comparing different countries to one another to discover how different climates and climate change might be connected to global health issues.

## Elevator Pitch:

Globally, climate change is becoming more unpredictable. Populations are not just experiencing extreme weather, but long-term health impacts that are linked to changes in the climate. Climate change is a gradual issue that is progressing every day, so the effects aren’t always immediately tangible. Over the past 11 years, data was collected from 25 different countries to create a dataset of over 14,000 weekly records of climate indicators, socioeconomic variables, and health outcomes like respiratory and cardiovascular disease rates. Our goal is to identify and visualize which environmental factors have the strongest relationship to global health impacts.

For our final product, we plan to build an interactive world map where users can click on a country and instantly see predicted health risks based on climate conditions, along with more data on the climate and socioeconomic context. This tool will make it easy to compare different countries and regions and help people understand how climate change patterns can shape public health in recent years and into the future.
