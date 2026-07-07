import datetime

import numpy as np
import pytz
import torch


RAD_PER_DEG = torch.tensor(np.pi / 180.0)
DATETIME_2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc).timestamp()


def _dali_mod(a, b):
    return a - b * torch.floor(a / b)


def cos_zenith_angle(time, latlon):
    lat = latlon[0:1, :, :].unsqueeze(0) * RAD_PER_DEG
    lon = latlon[1:2, :, :].unsqueeze(0) * RAD_PER_DEG
    time = time.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return _star_cos_zenith(time, lat, lon)


def _days_from_2000(model_time):
    return (model_time - DATETIME_2000) / (24.0 * 3600.0)


def _greenwich_mean_sidereal_time(model_time):
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    )
    return _dali_mod((theta / 240.0) * RAD_PER_DEG, 2 * np.pi)


def _local_mean_sidereal_time(model_time, longitude):
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _sun_ecliptic_longitude(model_time):
    julian_centuries = _days_from_2000(model_time) / 36525.0
    mean_anomaly = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    ) * RAD_PER_DEG
    mean_longitude = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    ) * RAD_PER_DEG
    d_l = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * torch.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * torch.sin(2 * mean_anomaly)
        + 0.000290 * torch.sin(3 * mean_anomaly)
    ) * RAD_PER_DEG
    return mean_longitude + d_l


def _obliquity_star(julian_centuries):
    return (
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries**2)
            + 0.00200340 * (julian_centuries**3)
            - 0.576e-6 * (julian_centuries**4)
            - 4.34e-8 * (julian_centuries**5)
        )
        / 3600.0
    ) * RAD_PER_DEG


def _right_ascension_declination(model_time):
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)
    eclon = _sun_ecliptic_longitude(model_time)
    x = torch.cos(eclon)
    y = torch.cos(eps) * torch.sin(eclon)
    z = torch.sin(eps) * torch.sin(eclon)
    r = torch.sqrt(1.0 - z * z)
    declination = torch.atan2(z, r)
    right_ascension = 2 * torch.atan2(y, (x + r))
    return right_ascension, declination


def _local_hour_angle(model_time, longitude, right_ascension):
    return _local_mean_sidereal_time(model_time, longitude) - right_ascension


def _star_cos_zenith(model_time, lat, lon):
    ra, dec = _right_ascension_declination(model_time)
    h_angle = _local_hour_angle(model_time, lon, ra)
    return torch.sin(lat) * torch.sin(dec) + torch.cos(lat) * torch.cos(dec) * torch.cos(h_angle)
