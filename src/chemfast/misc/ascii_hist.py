import numpy as np

import numpy as np


def print_ascii_histogram(
        x: np.ndarray,
        y: np.ndarray,
        max_width: int = 50,
        max_lines: int = 15,
        symbol: str = '█',
        fmt: str = 'd'
):
    if len(y) == 0:
        return "Empty Data!"

    # 辅助格式化函数，处理格式化浮点数为整数 'd' 时可能抛出的异常
    def _format_val(val, fmt_str):
        if 'd' in fmt_str:
            return f"{int(val):{fmt_str}}"
        return f"{val:{fmt_str}}"

    if len(x) == len(y) + 1:
        labels = [f"[{_format_val(x[i], fmt)}, {_format_val(x[i + 1], fmt)})" for i in range(len(y))]
    elif len(x) == len(y):
        labels = [f"{_format_val(val, fmt)}" for val in x]
    else:
        raise ValueError(f"Dim Mismatch: len(x)={len(x)}, len(y)={len(y)}")

    non_zero_idx = np.where(y > 0)[0]
    if len(non_zero_idx) == 0:
        return "All Data are 0!"

    if len(non_zero_idx) > max_lines:
        top_indices = sorted(non_zero_idx, key=lambda i: y[i], reverse=True)[:max_lines]
        selected_idx = sorted(top_indices)
    else:
        selected_idx = non_zero_idx

    max_label_len = max(len(labels[i]) for i in selected_idx)
    max_y = np.max(y[selected_idx])

    ret = "+" + "=" * (max_label_len + 3 + max_width + 10 - 2) + '+' + '\n'
    ret += f"{'Bin':>{max_label_len}} | Count\n"
    ret += "+" + "-" * (max_label_len + 3 + max_width + 10 - 2) + '+' + '\n'

    prev_i = None
    for i in selected_idx:
        if prev_i is not None and i > prev_i + 1:
            gap_size = i - prev_i - 1
            skipped_counts = np.sum(y[prev_i + 1:i])

            if skipped_counts == 0:
                gap_text = f" +++ Skip {gap_size} Null bins +++"
            else:
                gap_text = f" +++ Hidden {gap_size} bins (hidden freq: {skipped_counts}) +++"

            ret += f"{'...':>{max_label_len}} | {gap_text}\n"

        count = y[i]
        if max_y > 0:
            bar_length = max(1, int((count / max_y) * max_width))
        else:
            bar_length = 0

        bar = symbol * bar_length
        ret += f"{labels[i]:>{max_label_len}} | {bar} ({count})\n"

        prev_i = i

    ret += "+" + "=" * (max_label_len + 3 + max_width + 10 - 2) + '+' + '\n'

    return ret