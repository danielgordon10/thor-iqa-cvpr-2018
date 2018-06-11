import time

def get_time_str():
    tt = time.localtime()
    time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' %
            (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
    return time_str


def encode(string, encoding='utf-8'):
    return string.encode(encoding)


def decode(string, encoding='utf-8'):
    return string.decode(encoding)
