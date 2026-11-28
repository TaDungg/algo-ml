"""Pipeline dữ liệu W2-S5: download -> clean -> resample -> save -> báo cáo.

Chạy toàn bộ bằng 1 lệnh (từ thư mục algo-ml):
    python -m src.data

Thư mục:
    data/raw/cache/   file thô từng năm (cache, tải đứt giữa chừng thì chạy lại sẽ tiếp)
    data/clean/       {SYMBOL}_{side}_{H1|H4|D1}.parquet
    reports/          data_quality.md + data_summary.csv
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import dukascopy_python.compat as duka
from dukascopy_python.compat.instruments import (
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_NZD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_USA500_IDX_USD,
    INSTRUMENT_USATECH_IDX_USD,
)

# ---------------------------------------------------------------- cấu hình
ROOT = Path(__file__).resolve().parents[1]      # thư mục algo-ml
RAW_DIR = ROOT / "data" / "raw" / "cache"
CLEAN_DIR = ROOT / "data" / "clean"
REPORT_DIR = ROOT / "reports"

NY = "America/New_York"
START_YEAR = 2012
END_YEAR = 2025          # chỉ năm đã kết thúc -> cache cố định -> chạy lại ra y hệt

# mã: (mã Dukascopy, đơn vị spread, loại lịch giao dịch)
#   đơn vị spread: FX = pip (0.0001; JPY = 0.01); chỉ số = điểm (1.0)
SYMBOLS = {
    "EURUSD": (INSTRUMENT_FX_MAJORS_EUR_USD, 0.0001, "fx"),
    "GBPUSD": (INSTRUMENT_FX_MAJORS_GBP_USD, 0.0001, "fx"),
    "USDJPY": (INSTRUMENT_FX_MAJORS_USD_JPY, 0.01, "fx"),
    "USDCHF": (INSTRUMENT_FX_MAJORS_USD_CHF, 0.0001, "fx"),
    "AUDUSD": (INSTRUMENT_FX_MAJORS_AUD_USD, 0.0001, "fx"),
    "USDCAD": (INSTRUMENT_FX_MAJORS_USD_CAD, 0.0001, "fx"),
    "NZDUSD": (INSTRUMENT_FX_MAJORS_NZD_USD, 0.0001, "fx"),
    "USA500": (INSTRUMENT_USA500_IDX_USD, 1.0, "index"),
    "USATECH": (INSTRUMENT_USATECH_IDX_USD, 1.0, "index"),
}
SIDES = {"bid": duka.OFFER_SIDE_BID, "ask": duka.OFFER_SIDE_ASK}
OHLCV = ["open", "high", "low", "close", "volume"]

MAX_REMOVED_PCT = 1.0    # clean() bỏ quá 1% số dòng -> dừng hẳn, không chạy tiếp lặng lẽ


# ---------------------------------------------------------------- download
def download(symbol, side, start=START_YEAR, end=END_YEAR, raw_dir=RAW_DIR):
    """Tải nến H1 theo từng năm, có cache. Trả về DataFrame index UTC, cột OHLCV.

    Năm đã có file cache thì đọc file, không tải lại. Năm chưa kết thúc thì
    tải nhưng KHÔNG cache (dữ liệu năm đó còn thay đổi).
    """
    instrument = SYMBOLS[symbol][0]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now(timezone.utc).year

    parts = []
    for year in range(start, end + 1):
        path = raw_dir / f"{symbol}_{side}_H1_{year}.parquet"
        if path.exists():
            part = pd.read_parquet(path)
        else:
            part = duka.fetch(
                instrument,
                duka.INTERVAL_HOUR_1,
                SIDES[side],
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            )
            part = part[OHLCV]
            if year < this_year:
                part.to_parquet(path)
        if len(part) > 0:
            parts.append(part)

    if not parts:
        raise ValueError(f"{symbol} {side}: không có dữ liệu {start}-{end}")
    return pd.concat(parts)


# ---------------------------------------------------------------- kiểm tra chất lượng
def _closed_mask(times, kind):
    """True = giờ thị trường đóng (không kỳ vọng có nến). times: DatetimeIndex UTC.

    FX: đóng từ 17:00 NY thứ Sáu tới 17:00 NY Chủ nhật.
    Chỉ số Mỹ (GIẢ ĐỊNH, kiểm lại bằng dữ liệu): như FX + nghỉ 17:00-18:00 NY mỗi ngày.
    Ngày lễ không được loại -> nằm trong 'missing'.
    """
    ny = times.tz_convert(NY)
    dow, hour = ny.day_of_week, ny.hour
    closed = ((dow == 4) & (hour >= 17)) | (dow == 5) | ((dow == 6) & (hour < 17))
    if kind == "index":
        closed = closed | (hour == 17)
    return closed


def quality_report(df, k=8, kind="fx"):
    """Tóm tắt lỗi dữ liệu. Không in, không sửa df người gọi."""
    idx = df.index
    full = pd.date_range(idx.min(), idx.max(), freq="h")
    missing = full.difference(idx)
    missing = missing[~_closed_mask(missing, kind)]

    prices = df[["open", "high", "low", "close"]]
    bad_price = (prices <= 0).any(axis=1)
    bad_ohlc = (
        (df["high"] < prices.max(axis=1)) | (df["low"] > prices.min(axis=1))
    )

    ret = df["close"].pct_change()
    z = ret.abs() / ret.rolling(24).std().shift(1)

    expected = len(df) + len(missing)
    return pd.Series({
        "rows": len(df),
        "duplicated": int(idx.duplicated().sum()),
        "missing": len(missing),
        "missing_pct": 100 * len(missing) / expected,
        "bad_price": int(bad_price.sum()),
        "bad_ohlc": int(bad_ohlc.sum()),
        "spikes": int((z > k).sum()),
    })


# ---------------------------------------------------------------- clean
def clean(df, max_removed_pct=MAX_REMOVED_PCT):
    """Trả về (df_sạch, log). Không sửa df người gọi.

    Quy ước:
      - sắp xếp theo thời gian, bỏ timestamp trùng (giữ bản đầu)
      - bỏ nến giá <= 0 hoặc OHLC vô lý
      - KHÔNG điền nến thiếu, KHÔNG xóa spike: chỉ ghi vào báo cáo
        (spike EURUSD 2025 đều rơi vào giờ tin/London mở -> là giá thật)
      - bỏ quá max_removed_pct % -> raise, vì chỉ số tính trên phần còn lại sẽ lệch
    """
    out = df[OHLCV].copy()
    if out.index.tz is None:
        raise ValueError("index phải có múi giờ")
    out.index = out.index.tz_convert("UTC")
    n0 = len(out)

    out = out.sort_index()
    dup = out.index.duplicated(keep="first")
    out = out[~dup]

    prices = out[["open", "high", "low", "close"]]
    bad = (
        (prices <= 0).any(axis=1)
        | prices.isna().any(axis=1)
        | (out["high"] < prices.max(axis=1))
        | (out["low"] > prices.min(axis=1))
    )
    out = out[~bad]

    removed = n0 - len(out)
    log = {"rows_in": n0, "duplicated_removed": int(dup.sum()),
           "bad_removed": int(bad.sum()), "rows_out": len(out)}
    if 100 * removed / n0 > max_removed_pct:
        raise ValueError(f"clean() bỏ {removed}/{n0} dòng (> {max_removed_pct}%): {log}")
    return out, log


# ---------------------------------------------------------------- resample
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
_RULES = {"H4": "4h", "D1": "D"}


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Gộp nến H1 thành H4/D1 theo ngày giao dịch FX (bắt đầu 17:00 New York).

    Output: index = giờ MỞ nến (UTC); thêm cột n_bars = số nến H1 trong nến lớn.
    Close của nến D1 chỉ biết được lúc nến ĐÓNG -> ghép vào H1 phải shift(1) ở D1.
    Không sửa df của người gọi.
    """
    if tf not in _RULES:
        raise ValueError(f"tf phải là một trong {list(_RULES)}, nhận {tf!r}")
    if df.index.tz is None:
        raise ValueError("index phải có múi giờ (tz-aware)")

    # Đồng hồ giao dịch: 17:00 NY -> 00:00. Bỏ tz để resample không vướng DST
    # (DST đổi lúc 2h sáng Chủ nhật NY, thị trường đang đóng).
    shifted = df.index.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)
    tmp = df.set_index(shifted)                 # bảng mới, df gốc giữ nguyên

    agg = {c: f for c, f in _AGG.items() if c in tmp.columns}
    grouped = tmp.resample(_RULES[tf])
    out = grouped.agg(agg)
    out["n_bars"] = grouped.size()
    out = out[out["n_bars"] > 0]                # bỏ ô cuối tuần/ngày lễ trống

    out.index = (out.index - pd.Timedelta(hours=7)).tz_localize(NY).tz_convert("UTC")
    out.index.name = df.index.name
    return out


# ---------------------------------------------------------------- spread
def spread_stats(bid, ask, unit):
    """Spread close_ask − close_bid theo đơn vị của mã (pip hoặc điểm)."""
    m = bid[["close"]].join(ask[["close"]], how="inner", lsuffix="_bid", rsuffix="_ask")
    spread = (m["close_ask"] - m["close_bid"]) / unit
    return pd.Series({
        "matched_rows": len(m),
        "spread_mean": spread.mean(),
        "spread_median": spread.median(),
        "negative_spread": int((spread < 0).sum()),
    })


# ---------------------------------------------------------------- save
def save(df, symbol, side, tf, clean_dir=CLEAN_DIR):
    path = Path(clean_dir) / f"{symbol}_{side}_{tf}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------- báo cáo
def _md_table(table):
    """DataFrame -> bảng markdown (không cần gói tabulate)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        return str(v)
    cols = [table.index.name or ""] + list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        lines.append("| " + " | ".join([str(name)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def write_report(table, start, end, report_dir=REPORT_DIR):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "data_summary.csv")
    text = f"""# Chất lượng dữ liệu ({start}–{end}, H1, nguồn Dukascopy)

Tạo tự động bởi `python -m src.data`. Không sửa tay.

{_md_table(table)}

## Cột
- `start`, `end`: nến H1 (bid) đầu và cuối; `rows_h1/h4/d1`: số nến bid sau clean.
- `missing_pct`: % giờ thiếu / số giờ kỳ vọng (lịch FX; chỉ số Mỹ thêm giờ nghỉ 17:00 NY — giả định). Ngày lễ tính là thiếu.
- `removed`: số dòng clean() bỏ (trùng + giá/OHLC vô lý), bid + ask.
- `spikes`: |ret| > 8 × std 24 nến trước (chỉ đánh dấu, không xóa).
- `spread_mean`: TB close_ask − close_bid; FX tính bằng pip (JPY 0.01), chỉ số tính bằng điểm.

## Quy ước clean
- Không điền nến thiếu, không xóa spike. Bỏ trùng (giữ bản đầu) và nến giá/OHLC vô lý.
- clean() bỏ quá {MAX_REMOVED_PCT}% số dòng thì dừng pipeline.
- D1/H4 cắt theo ngày giao dịch bắt đầu 17:00 New York; nhãn = giờ mở nến (UTC).
"""
    path = report_dir / "data_quality.md"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- chạy toàn bộ
def run_all(symbols=None, start=START_YEAR, end=END_YEAR):
    symbols = symbols or list(SYMBOLS)
    rows = {}
    for symbol in symbols:
        _, unit, kind = SYMBOLS[symbol]
        data, removed = {}, 0
        for side in SIDES:
            print(f"{symbol} {side}: tải/đọc cache {start}-{end} ...", flush=True)
            raw = download(symbol, side, start, end)
            df, log = clean(raw)
            removed += log["duplicated_removed"] + log["bad_removed"]
            data[side] = df
            save(df, symbol, side, "H1")
            for tf in _RULES:
                save(resample_ohlc(df, tf), symbol, side, tf)

        bid = data["bid"]
        q = quality_report(bid, kind=kind)
        sp = spread_stats(bid, data["ask"], unit)
        rows[symbol] = {
            "start": bid.index.min(),
            "end": bid.index.max(),
            "rows_h1": len(bid),
            "rows_h4": len(resample_ohlc(bid, "H4")),
            "rows_d1": len(resample_ohlc(bid, "D1")),
            "missing_pct": q["missing_pct"],
            "removed": removed,
            "spikes": int(q["spikes"]),
            "spread_mean": sp["spread_mean"],
            "negative_spread": int(sp["negative_spread"]),
        }
        print(f"{symbol}: xong ({len(bid)} nến H1)", flush=True)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "symbol"
    path = write_report(table, start, end)
    print(f"Báo cáo: {path}")
    return table


if __name__ == "__main__":
    run_all()
