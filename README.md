# <a name="top"></a>ZILLOW DATASET 
![]()

by: Alfred W. Pirovits

<p>
  <a href="https://github.com/Alfred-W-S-Pirovits-Jr/telco_churn_project#top" target="_blank">
    <img alt="" src="" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

<img src="https://docs.google.com/drawings/d/e/2PACX-1vR19fsVfxHvzjrp0kSMlzHlmyU0oeTTAcnTUT9dNe4wAEXv_2WJNViUa9qzjkvcpvkFeUCyatccINde/pub?w=1389&amp;h=410">

## <a name="project_goals"></a> Project Goals:
- Explore 10 csv's of data including information on airline data from 2009 to 2019 for the flights in North America
- Make sense of the data nnd try and see if I can use it to discover underlying patterns in airline delays both at the Airline Level and at the National Airspace level
- Clean the data to only use time information after which, if time permits, maybe add in other features.
- Try and create a viable model that can predict airline delays based on historical trends

## <a name="project_description"></a>Project Description:
[[Back to top](#top)]
The purpose of this project is to look at all of the massive amounts of data and see if I can garner greneral trends that may prove useful to the general aviation community.  I suspect that there is a yearly pattern that holds and dictates delays given the four seasons in a year but it would be nice to show that there is a repeatable trend.  Also I am wondering if these results will be different by major airlines as they often own different hubs in the transportations.  Different airports have diffenent airlines operating out of them as main hubs.
***
## <a name="planning"></a>Project Planning:    
[[Back to top](#top)]
The main goal of the project was to explore the data presented and see what I could discover.  Since there is a lot of data to go through I want to ct it down into a manageable set of features that I could use to characterize delays.  I am relying on domain knowledge as a holder of a Commercial Pilot's License to choose initial features as appropriate.  After doing this, I wanted to initially choose a representative airline to do the initial model.  United Airlines is a great choice since it's main hubs span the entire system and covers most of the major regions in the United States National Airspace Syste.



        
### Hypothesis
The main hypothesis is that there would be a meaningful seasonal perodicity to the data as the seasons correlate to delays year over year.  I was also suspecting  



### Target variable
The Target Variable is Total minutes delayed coded as average_delay in the resampled dataframe.  I totaled up the 5 types of delays into one column.  These were Carrier Delay, Weather Delay, NAS Delay (National Airspace System), Security Delay and Late Aircraft Delay which were combined into a total delay column.  Then the dataframe was resampled to mean delay by day and further analyzed.  In the end I did a time series analysis on the delay averaged over two week periods.   


### Need to haves (Deliverables):
Github repo with the following:

1. Readme (.md)
- Project goals
- Project description
- Project planning (lay out your process through the data science pipeline)
- Initial hypotheses and/or questions you have of the data, ideas
- Data dictionary
- Instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
- Key findings, recommendations, and takeaways from your project.


2. Acquire & Prepare Modules (.py) (COMBINED INTO A WRANGLE AS WELL)
- Contains functions to acquire, prepare and split your data. You can have other .py files if you desire to abstract other code away from your final report.
- Each of your functions are accompanied by descriptive docstrings. If they are functions you borrowed from instructors, put those docstrings in your own words.
- Functions to acquire and prepare your data should be imported and used in your final report.
- Your work must be reproducible by someone with their own env.py file.

3. Final Report (.ipynb)
- A Report that has filtered out all the extraneous elements not necessary to include in the report.
- Use markdown throughout the notebook to guide the audience. Assume the reader will not read your code blocks as you think about how much markdown guidance do you need.
- Then, assume another reader will read ALL of your code, so make sure it is clearly commented. All cells with code need comments.
- Your notebook should begin with a project overview and goals
- Exploration should be refined in the report because now you know which visualizations and tests led to valuable outcomes.
- Include at least 4 visualizations in the form of:
    Question in markdown that you want to answer
    Visualization
    Statistical test (in at least 2 of your 4)
    Provide your clear answer or takeaway in markdown and natural language to the question based on your exploration.
- Provide the context of the target variable through a visualization (distribution of the values, e.g.)
- Include your 3 best models in the final notebook to review. Show the steps and code you went through to fit the models, evaluate, and select.
- On your best model, a chart visualizing how it performed on test would be valuable.
- End with a conclusion that talks about your original goals and how you reached those (or didn't), the key findings, recommendations and next steps ("If I had more time, I would...")

4. non-final Notebook(s) (.ipynb)
Additional non-final notebooks (.ipynb) may be created while working on the project, containing exploration, other work, or modeling work, but they will not be graded. All required elements must be in the final report notebook.



### Nice to haves (With more time):
With more time I might further divide this models by a given hub.  At this point the notebook can be run for any airline with continuous data.  This will include all the major airlines like American, Southwest, Delta, United and others.  I would like to see if further breaking down airline trends by hub would be useful.  It may or may not as usually airlines dedicate a tail number to a 3 leg trip and it remains consistent. 

Also, I would like to explore if I could use the Holt's Seasonal Trend model as a baseline from which I can and hone in on the errors with more information.  For instance if the Seasonal model shows an average delay of 10 minutes but it was 45, what accounts for the difference.  My inclination is that weather is the biggest factor.  However, even though year over year trends in weather are somewhat predictable, the day to day variation is not.  Christmas one year might be a blizzard in New York, but the next might be clear skies.  Yet it usually snows sometime in the winter.  I would probably feature engineer the difference between the average_delay predicted by this model and the actual delay for each flight and find features that may have an impact and probably run a linear regression on it.  Then I would combine the two to create a better prediction.

I also would like to see if I can get the METAR reports for those 10 years.  Every airport disseminates weather conditions to pilots once an hour.  This data has to be held somewhere.  I would like to see if I could find and append this data to the already large dataset to see if that helps.  That data would be linked by departure and arrival airports and times of departure and arrival.




***

## <a name="findings"></a>Key Findings, Recommendations and Takeaways:
[[Back to top](#top)]
The key finding is that with an RMSE of 3.74 minutes which beat the best baseline of .55 minutes, there isn't much value in the actual model by the fortnight to predict actual delays.  However the seasonal trend is absolutely valuable.  It formalizes what the airlines already know.  That there is a seasonality to the delays.  But interestingly enough, the biggest average delay peaks are happening in the summertime versus the wintertime which is the exact opposite of my expectations.  Perhaps the airlines have accounted for the winter delays but have not sufficiently accounted for the summer delays.  

I recommend putting the seasonal trend data against known delays and see if there is an agreement or an adjustment that needs to be made.  I also note that this is an MVP and though almost useless as a practical model, it may still prove to turn into something promising.  I made the error of combining the delays at the beginning of this project due to time constraints.  If I were to do it again I would do multiple models...one for each target and see if this comes up with more promising results.  

As a takeaway, I learned a bit about the time series analysis and think that I might use more models like FP prophet to get a better handle of this data.  Also since Holt's Seasonal trend with damping worked best on test and validate, I wonder if its better to use the non damped data as a basis for the ensemble method that I suggested above.  

***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]
FL_DATE OP_CARRIER CARRIER_DELAY WEATHER_DELAY NAS_DELAY SECURITY_DELAY LATE_AIRCRAFT_DELAY
### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| FL_DATE | The date of the flight in question (initially object but converted) | datetime64 |
| OP_CARRIER | Two letter IATA carrier code for the airline in question in minutes | object |
| CARRIER_DELAY | Delay caused by the carrier in minutes | object |
| WEATHER_DELAY | The assessed tax value of the home in minutes | object |
| NAS_DELAY | NAS (national airspace system) delay caused in minutes | object |
| SECURITY_DELAY | Delay caused by security problems in minutes | object |
| LATE_AIRCRAFT_DELAY | Delay caused by aircraft coming in late from previous flight in minutes | object |
| total_delay | The total of all the delay columns created during feature engineering in minutes | float64 | 
| average_delay | Average of the total delay column resampled by duration in minutes | float64 |

***


## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.ipynb
    - wrangle.py
    - explore.py
    - explore.ipynb
    - model.py
    - model.ipynb
    - final_notebook.ipynb

The steps to look through the MVP are in the final notebook.  There are a lot of functions in the preliminary  exploration, the acquire and the prepare files that one can use to explore further but for the purposes of reproducing this mvp all that is needed is in the wrangle.py file and the project_final.ipynb.
### Takeaways from exploration:


***

## <a name="Conclusion"></a>Conclusion:
[[Back to top](#top)]
This is a long way away from a viable real world product.  However it is a very good start.  More work is needed to seperate the causes of delays and group them in a more meaningful way.  This has been an excellent project to learn time series models on.  I would want to create an ensemble model in a later iteration of this project.  All in all:
    - There is a clear seasonality to the delays
    - Our rmse is just over a minute better than basline but really is not all that useful in and of itself
    - The seasonal trend IS useful and can inform expected delays given the time of the year
    - There are a lot of residuals...more on that below in next steps
    - Even the best models couldn't predict COVID!!!