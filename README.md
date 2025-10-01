[![Python CI for IDS706 (bos_temp)](https://github.com/yuqianw2002/week2_mini_prj_yw/actions/workflows/main.yml/badge.svg)](https://github.com/yuqianw2002/week2_mini_prj_yw/actions/workflows/main.yml)

# Week2_mini_prj_yw - Analysis Boston weather from 2013 to 2023

## Project Description 
This project analyzes Boston weather data from 2013-2023, focusing on temperature prediction using weather features (wind speed, pressure, wind direction). After group the data with each month, then filter the winter monthes which from 2013 to 2023 to compare the differences of all weather data. To indicate of the trend of winter. Use linear regression model on the whole data to predict future temperature. The scatter plot have the red dot as the predition temperature, and blue is the real tempearture. X aixs is Wind speed and Y axis is temperature.  

## Set up
1. Download the Boston weather from 2013 to 2023 dataset from Kaggle: https://www.kaggle.com/datasets/swaroopmeher/boston-weather-2013-2023/data
2. Save the csv file in the project folder. 
- This dataset contains the following columns:
    - time: The date in string format
    - tavg: The average air temperature in Celsius, as a float
    - tmin: The minimum air temperature in Celsius, as a float
    - tmax: The maximum air temperature in Celsius, as a float
    - prcp: The daily precipitation total in millimeters, as a float
    - wdir: The average wind direction in degrees, as a float
    - wspd: The average wind speed in kilometers per hour, as a float
    - pres: The average sea-level air pressure in hectopascals, as a float
3. Run Makefile to install all the packages using in for the projects.

## Result
From the "test_pred.png", the predict temperature and the actual temperature are overlapping together, and the mean squared error of the predcit data and actual data is around 73.54. By using the simple lineat regression model to predic the temperature for Boston weather, it have a relevently good accuracy. 

## Dev-container
1. Press `Cmd + Shift + P`, then click on "Dev Containers: Add Dev Container Configuration Files". 
2. Select the configuration of choice. Because we use all Python code, I choose Python 3. 
3. In the next menu, add more features as needed. 
4. Open the Dev container by clicking on the blue button on the bottom left corner of VSCode. 

## Docker
1. Press `Cmd + Shift + P`, then click on "Containers: Add Docker Files to Workspace." 
2. In the terminal, ran: `docker build -t container-name .` I use 'bos_weather' as the image name.
3. Next, I ran `docker run -d -p 8088:3000 --name bos-weather-container bos_weather`. This run the container in the Docker dashboard. 







