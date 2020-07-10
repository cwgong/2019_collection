# -*- encoding:utf8 -*-
import time
import datetime

def n_days_ago_milli_time(n):
    return int(round((time.time() - n * 24 * 60 * 60) * 1000))


def current_milli_time():
    return int(round(time.time() * 1000))
