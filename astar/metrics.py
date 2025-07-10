import csv, time
def Logger(path):
    f = open(path, "w", newline="")
    wr = csv.writer(f); wr.writerow(["algo","map_seed","len","expansions","cpu_ms"])
    def log(*row):
        wr.writerow(row); f.flush()
    return log
