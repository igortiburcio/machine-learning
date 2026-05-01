import csv
from math import log2

def load_csv(path: str) -> tuple[list[dict[str, str]], list[str], str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = list(rows[0].keys())
    return rows, headers[:-1], headers[-1]

def start() -> None:
    data, attributes, target = load_csv("data/risco_credito.csv")

    initial_entropy = calculate_entropy(data, target)
    print(f"Entropia inicial: {initial_entropy:.4f}")

    for attr in attributes:
        gain = calculate_information_gain(data, attr, target)
        print(f"Ganho de informação ({attr}): {gain:.4f}")

def calculate_entropy(data: list, target: str) -> float:
    targets = [row[target] for row in data]
    total = len(targets)

    if total == 0:
        return 0
    
    count: dict[str, int] = {}
    for target in targets:
        count[target] = count.get(target, 0) + 1

    entropy = 0.0
    for freq in count.values():
        probability = freq / total
        entropy -= probability * log2(probability)

    return entropy

def calculate_information_gain(data: list, attribute: str, target: str) -> float:
    values = set(row[attribute] for row in data)
    subsets = [[row for row in data if row[attribute] == val] for val in values]
    initial_entropy = calculate_entropy(data, target)

    gain = 0

    for subset in subsets:
        subset_entropy = calculate_entropy(subset, target)
        weight = len(subset) / len(data)
        gain += weight * subset_entropy

    return  initial_entropy - gain