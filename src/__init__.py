rows = [
    ("EURUSD", "1.08500", "1.08900"),
    ("GBPUSD", "1.27x00", "1.27650"),   # gõ nhầm
    ("USDJPY", "14x.200", "147.650"),
]

good, bad = [], []
for pair, entry_s, exit_s in rows:
    try:
        entry = float(entry_s)
        exit_ = float(exit_s)          # nếu dòng trên lỗi, dòng này KHÔNG chạy
    except ValueError:
        # ghi lại để sửa, không dừng chương trình
        bad.append(pair)
        continue                       # sang dòng tiếp theo
    good.append((pair, entry, exit_))

print(len(good), "dòng hợp lệ; dòng lỗi:", bad)
