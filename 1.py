import zstandard as zstd
import json
import io

with open("feedbacks-02.json.zst", "rb") as f:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(f) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        for i, line in enumerate(text_stream):
            review = json.loads(line)
            print(review)

            if i == 9:
                break