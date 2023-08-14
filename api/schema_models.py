from pydantic import BaseModel, root_validator, validator
from datetime import date
from enum import Enum
import re


class IntervalField(str, Enum):
    DAILY = 'daily'
    TIME_INTERVAL = 'time_interval'


class ApiPostData(BaseModel):
    location_id: int
    date: str

    @validator('date')
    def validate_passwords_match(cls, value):
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        return value


class TrainModel(BaseModel):
    intervals: IntervalField
    max_date: str

    @validator('max_date')
    def validate_passwords_match(cls, value):
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        return value

