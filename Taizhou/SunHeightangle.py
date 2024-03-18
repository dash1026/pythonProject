from datetime import datetime, timedelta
import math

# Define the latitude for Taizhou, Zhejiang, China. This is an approximate value.
latitude = 28.6564  # North

# The date for which to calculate. Assuming today's date for the calculation.
# Note: The user didn't specify the date, so we assume the current date.
# My knowledge is updated until April 2023, and I'll use a date close to that for this calculation.

date = datetime(2018, 4, 27)  # Assuming a date in spring for the example.

# Calculate the day of the year (N)
day_of_year = date.timetuple().tm_yday

# Calculate the approximate time of solar noon at the location in question.
# Equation of time (in minutes)
EoT = 9.87 * math.sin(2 * math.pi * (day_of_year - 81) / 364) - 7.53 * math.cos(math.pi * (day_of_year - 81) / 364) - 1.5 * math.sin(math.pi * (day_of_year - 81) / 364)

# Standard meridian for the timezone (China Standard Time, UTC+8)
standard_meridian = 120

# Longitude of the location (Taizhou, Zhejiang, China). This is an approximate value.
longitude = 121.4208  # East

# Time correction factor (in minutes)
TC = 4 * (longitude - standard_meridian) + EoT

# Local solar noon (in hours)
LSN = 12 + TC / 60

# Calculate the solar declination angle (in radians)
# Solar declination delta (in degrees)
delta = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

# Convert solar declination angle to radians
delta_rad = math.radians(delta)

# Convert latitude to radians
latitude_rad = math.radians(latitude)

# Calculate the hour angle (in degrees) at 19:35 (7:35 PM) local time.
# Time difference from local solar noon, in hours
time_diff = 19 + 35 / 60 - LSN

# Hour angle, in degrees
H = time_diff * 15  # 15 degrees per hour

# Convert hour angle to radians
H_rad = math.radians(H)

# Calculate the solar altitude angle (alpha) in degrees
# sin(alpha) = sin(delta) * sin(latitude) + cos(delta) * cos(latitude) * cos(H)
alpha_rad = math.asin(math.sin(delta_rad) * math.sin(latitude_rad) + math.cos(delta_rad) * math.cos(latitude_rad) * math.cos(H_rad))

# Convert solar altitude angle from radians to degrees
alpha = math.degrees(alpha_rad)

print(alpha)

