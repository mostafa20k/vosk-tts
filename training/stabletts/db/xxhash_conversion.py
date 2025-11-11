import pickle
import xxhash

input_path = "dictionary.pkl"
output_path = "dictionary_xxhash.pkl"

with open(input_path, "rb") as f:
    wdic, probs = pickle.load(f)

# Convert keys to xxHash64 hashes
hashed_wdic = {}
hashed_probs = {}

for word, value in wdic.items():
    h = xxhash.xxh64(str(word).encode("utf-8")).intdigest()
    hashed_wdic[h] = value

for word, prob in probs.items():
    h = xxhash.xxh64(str(word).encode("utf-8")).intdigest()
    hashed_probs[h] = prob

with open(output_path, "wb") as f:
    pickle.dump((hashed_wdic, hashed_probs), f)

print(f"✅ Converted {len(wdic):,} entries to xxHash and saved to {output_path}")
