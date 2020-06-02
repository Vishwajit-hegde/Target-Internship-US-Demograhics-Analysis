# US-Demograhics-Analysis
This analysis is a part of my internship at Target Corp. \
The analysis is aimed at identifying different areas which have similar demographical features using clustering. 
## Web Scraping US demographics data
The required data is at 5-digit zip code level. There are totally 33120 zip codes. The data is available on US census bureau website. There are different data tables 
with data at different levels of granularity. The data sets are available with 27000 attributes, 18000 attributes and 1000 attributes.
I chose the one with around 1000 attributes. Data can be downloaded using the census api or directly from the website. 
The descriptions of these attributes are extracted using BeautifulSoup library.
## Attributes Analysis
The attributes descriptions are organised in a way which is helpful to understand them.
## Data Cleaning
Zip codes corresponding to Puerto Rico have a separate list of attributes. They are combined with the attributes of other zip codes
as I did not see a need for a separate set. Each attribute contains the estimate, percent estimate, margin error, percent margin error.
For the time being percent estimate is chosen. If some attribute doesn't have the percent value, actual value is chosen. After manually
going through all the attributes 238 important ones are chosen. The missing values are filled with median values.
## Visualizations
(Under progress)
Out of the chosen attributes, some are selected so that they can be visualized using pie charts, bar graphs etc.
Also the distributions of these attributes are visualized using density plots. 


