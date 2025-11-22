# Forecast Plan

## Prompt

I have about 400 gas stations that I need to forecast fuel volume for. I have daily sales data for the past four years, with the data separated by fuel type. I currently pull the data from PDI, which is the proprietary pricing site we use. The data comes out as an xlsx file. The column headers start on row 5 of the xlsx output. The sales history is too large to pull in one go, so I usually have to pull the data by year and then combine it. I'd like to create my own local database or something like that that I can continuously update as new sales data comes in. Ideally there could be a way to know I am not duplicating data or anything like that. I would then like to be able to create a monthly forecast for aggregate fuel demand. The final result needs to be montly volume by **store**. I also need the option to do it by fuel type or by specific store and fuel type if needed. We constantly have new stores opening and old stores closing; I am not sure how to best handle this but I would like it to be handled in a practical way. I will need the forecast at least two months before the month I am forecasting. I would like to use ETS/Holt-Winters and Seasonal Naive for now. I think python be best for this task. I want to use the uv python package manager.  Ideally I can evaluate the forecasting methods and see which ones perform the best and are the most robust.

My job requires that I forecast monthly fuel volume. I'd like to make this process as easy and simple and repeatable as possible.

Here are the columm names for the data:

- Brand, Site Id, Site, Address, City, State, Owner, B/Unit, Grade, Day, Stock, Delivered, Volume, Is Estimated, Total Sales, Target
