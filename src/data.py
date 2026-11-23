
import numpy as np
import pandas as pd


def quality_report(df, k):
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')

    # 1) Lịch đầy đủ: mỗi giờ 1 nến
    full = pd.date_range(
        start=idx.min(),
        end=idx.max(),
        freq='h'
    )

    # 2) Tìm timestamp bị thiếu
    missing = full.difference(idx)

    # 3) Loại bỏ weekend gaps theo giờ New York
    ny = missing.tz_convert('America/New_York')

    weekend = (
        ((ny.day_of_week == 4) & (ny.hour >= 17)) |
        (ny.day_of_week == 5) |
        ((ny.day_of_week == 6) & (ny.hour < 17))
    )

    missing_trading = missing[~weekend]

    print(f"There are: {len(missing_trading)} missing candles")
    print(missing_trading)

    # 4) Duplicated data
    duplicates = df.index.duplicated()

    if duplicates.sum() > 0:
        print("There are duplicated candles:")
        print(df[df.index.duplicated(keep=False)])
    else:
        print("There are no duplicated candles")

    # 5) Invalid candles
    invalid_price = (
        (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    )

    invalid_ohlc = (
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['high'] < df['low'])
    )

    invalid_candles = invalid_price | invalid_ohlc
    bad_rows = df[invalid_candles]

    print(f"Invalid price: {invalid_price.sum()}")
    print(f"Invalid OHLC: {invalid_ohlc.sum()}")
    print(f"Invalid candles: {len(bad_rows)}")
    print(bad_rows)

    # 6) Anomaly: Spiked candles
    ret = df['close'].pct_change()

    vol = ret.rolling(24).std().shift(1)
    z = ret.abs() / vol

    flag = df[z > k].copy()
    flag['dow'] = flag.index.day_of_week
    flag['hour'] = flag.index.hour

    print(flag)

    # 7) Summary report
    return pd.Series({
        'rows': len(df),
        'duplicated': duplicates.sum(),
        'missing': len(missing_trading),
        'bad_price': invalid_price.sum(),
        'bad_ohlc': invalid_ohlc.sum(),
        'spikes': len(flag)
    })
