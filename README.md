# API-Based-Weather-Forecasting-Model
This Model automates the process of importing data from weather, air quality, and financial markets for time series forecasting. It uses APIs for each domain and processes the data for time series analysis.

Air Pollution API concept -

Air Pollution API provides current, forecast and historical air pollution data for any coordinates on the globe.

Besides basic Air Quality Index, the API returns data about polluting gases, such as Carbon monoxide (CO), Nitrogen monoxide (NO), Nitrogen dioxide (NO2), Ozone (O3), Sulphur dioxide (SO2), Ammonia (NH3), and particulates (PM2.5 and PM10).

Air pollution forecast is available for 4 days with hourly granularity. Historical data is accessible from 27th November 2020.

Here is a description of OpenWeather scale for Air Quality Index levels:

Qualitative name	Index	Pollutant concentration in μg/m3
SO2	NO2	PM10	PM2.5	O3	CO
Good	1	[0; 20)	[0; 40)	[0; 20)	[0; 10)	[0; 60)	[0; 4400)
Fair	2	[20; 80)	[40; 70)	[20; 50)	[10; 25)	[60; 100)	[4400; 9400)
Moderate	3	[80; 250)	[70; 150)	[50; 100)	[25; 50)	[100; 140)	[9400-12400)
Poor	4	[250; 350)	[150; 200)	[100; 200)	[50; 75)	[140; 180)	[12400; 15400)
Very Poor	5	⩾350	⩾200	⩾200	⩾75	⩾180	⩾15400
Other parameters that do not affect the AQI calculation:

NH3: min value 0.1 - max value 200
NO: min value 0.1 - max value 100

Website - https://openweathermap.org/api/air-pollution
![image](https://github.com/user-attachments/assets/085b03de-f728-4e5b-bbbb-4852271d5027)
