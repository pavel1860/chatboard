from datetime import datetime, timedelta
import pytz


def get_int_timestamp(ref_time=None):
    if ref_time:
        return int(ref_time.timestamp() * 1000000)
    return int(datetime.now().timestamp() * 1000000)



def get_local_datetime(country_code: str):
    timezone_str = pytz.country_timezones[country_code]
    if not timezone_str:
        raise ValueError(f"Country code {country_code} not found")
    country_timezone = pytz.timezone(timezone_str[0])
    # country_time = country_timezone.localize(datetime.now())
    country_time = datetime.now(country_timezone)
    return country_time


def to_local_datetime(dt, country_code="UTC"):
    if country_code == "UTC":
        return dt.astimezone(pytz.utc)
    timezone_str = pytz.country_timezones[country_code]
    tz = pytz.timezone(timezone_str[0])
    return dt.astimezone(tz)


def to_utc_datetime(dt):
    return dt.astimezone(pytz.utc)


def is_in_time_window(time, from_time=None, hours=0, minutes=0, seconds=0):
    if from_time is None:
        from_time = datetime.now()
    start_time = from_time.astimezone(time.tzinfo)
    time_window = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    end_time = start_time - time_window    
    return end_time <= time <= start_time




def convert_datetime_timezone(origin: datetime, origin_code: str, target_code: str="UTC") -> datetime:
    """Converts a datetime object from one timezone to another timezone"""
    if origin.tzinfo is not None:
        origin = origin.replace(tzinfo=None)

    if origin_code == "UTC":
        origin_time = pytz.utc.localize(origin)
    else:
        origin_str = pytz.country_timezones[origin_code]
        origin_time_zone = pytz.timezone(origin_str[0])
        origin_time = origin_time_zone.localize(origin)    

    # Convert to UTC
    if target_code == "UTC":
        utc_time = origin_time.astimezone(pytz.utc)
        return utc_time
    else:
        target_str = pytz.country_timezones[target_code]
        target_time_zone = pytz.timezone(target_str[0])
        target_time = origin_time.astimezone(target_time_zone)
        return target_time
