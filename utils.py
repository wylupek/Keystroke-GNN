

def print_csv(content: list[dict]) -> None:
    print("key\tpress_time\t\tduration\t")
    for row in content:
        print(f"{row['key']}\t{row['press_time']}\t{row['duration']}")