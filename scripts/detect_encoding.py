import chardet

path = "data/raw/Tender24x7/archive.csv"

with open(path, "rb") as f:
    raw = f.read(200000)   # read first 200KB

result = chardet.detect(raw)
print(result)
