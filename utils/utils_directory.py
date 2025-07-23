import os

def get_next_exp_dir(base_dir="results", prefix="exp_"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
    numbers = [int(d.replace(prefix, "")) for d in existing if d.replace(prefix, "").isdigit()]
    next_num = max(numbers) + 1 if numbers else 0
    exp_dir = os.path.join(base_dir, f"{prefix}{next_num}")
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, "models"))
    os.makedirs(os.path.join(exp_dir, "images"))
    return exp_dir
