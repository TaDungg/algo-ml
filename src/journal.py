import csv
import os
import sys


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def calc_fx_pair(pair, directions, entry_price, exit_price, lot_size):
    pips = directions * (exit_price - entry_price) / pip_size(pair)
    if pair.startswith("USD"):
        pip_value = pip_size(pair) * 100000 / exit_price
    else:
        pip_value = pip_size(pair) * 100000
    profit = pips * pip_value * lot_size  # calculate the profit
    return profit


def win_rate(result_profit):
    win_count = 0
    for p in result_profit:
        if p > 0:
            win_count += 1
        else:
            pass
    trade_winrate = win_count / len(result_profit)
    return trade_winrate


def loss_rate(result_profit):
    loss_count = 0
    for p in result_profit:
        if p < 0:
            loss_count += 1
        else:
            pass
    trade_lossrate = loss_count / len(result_profit)
    return trade_lossrate


def win_avg(result_profit):
    total_win = 0
    win_count = 0
    for p in result_profit:
        if p > 0:
            total_win += p
            win_count += 1
        else:
            pass
    avg_win = total_win / win_count
    return avg_win


def loss_avg(result_profit):
    total_loss = 0
    loss_count = 0
    for p in result_profit:
        if p < 0:
            total_loss += p
            loss_count += 1
        else:
            pass
    avg_loss = total_loss / loss_count
    return avg_loss


def longest_losing_streak(result_profits):
    streaks = []
    current = 0
    for p in result_profits:
        if p < 0:
            current += 1
        else:
            streaks.append(current)
            current = 0
    streaks.append(current)
    return max(streaks)


def avg_rr(result_profit):
    return win_avg(result_profit) / abs(loss_avg(result_profit))


def avg_expectancy(result_profit):
    return sum(result_profit) / len(result_profit)


def expectancy_v2(result_profit):
    expectancy = win_rate(result_profit) * win_avg(result_profit) + \
        loss_rate(result_profit) * loss_avg(result_profit)
    return expectancy


def save_journal(trades, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "pair", "dir", "entry", "exit", "stop", "lot"])
        writer.writeheader()
        writer.writerows(trades)


def direction_sign(d):
    if d == "BUY":
        return 1
    elif d == "SELL":
        return -1
    else:
        raise ValueError(f"Hướng lệnh không hợp lệ: {d!r}")


def load_journal(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        reader = csv.DictReader(f)
        for row in reader:
            try:
                direction_sign(row["dir"])
                row["entry"] = float(row["entry"])
                row["exit"] = float(row["exit"])
                row["stop"] = float(row["stop"])
                row["lot"] = float(row["lot"])
            except ValueError as e:
                print(f"Bỏ qua dòng {reader.line_num}: {e}")
                continue
            rows.append(row)
    return rows


class Trade:
    def __init__(self, pair, direction, entry, exit, lot):
        self.pair = pair
        self.direction = direction
        self.entry = entry
        self.exit = exit
        self.lot = lot

    @property
    def sign(self):
        return direction_sign(self.direction)

    @property
    def pnl(self):
        return calc_fx_pair(self.pair, self.sign, self.entry, self.exit, self.lot)
