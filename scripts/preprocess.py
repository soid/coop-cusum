import gzip
import json
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path
import sys

import click
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import numpy as np

PRINTABLE = set(string.printable)

MIN_REV_LEN = 45
MAX_REV_LEN = 512


def strip_text(s: str) -> str:
    # https://stackoverflow.com/a/518232/2809427
    # https://stackoverflow.com/a/8689826
    return re.sub(" +", " ", "".join(c for c in unicodedata.normalize("NFD", s)
                                     if unicodedata.category(c) != "Mn" and c in PRINTABLE).replace("\n", " "))


def yelp(file_path: str, spm: SentencePieceProcessor = None):
    d = defaultdict(list)
    for ins in tqdm(map(json.loads, open(file_path)), desc="Yelp"):
        rating = int(ins["stars"])
        text = strip_text(ins["text"])
        x = {"business_id": ins["business_id"],
             "review_id": ins["review_id"],
             "rating": rating,
             "text": text}
        if spm is not None:
            piece = spm.Encode(text)
            if MIN_REV_LEN <= len(piece) <= MAX_REV_LEN:
                x["piece"] = piece
                d[ins["business_id"]].append(x)
        else:
            d[ins["business_id"]].append(x)

    for reviews in d.values():
        if len(reviews) > 10:
            yield from reviews


def amzn(dir_path: str, spm: SentencePieceProcessor = None):
    p = tqdm()
    obs = set()
    for fp in Path(dir_path).glob("*.gz"):
        p.set_description(desc=fp.stem)
        d = defaultdict(list)
        for ins in filter(lambda x: x["asin"] not in obs, map(json.loads, gzip.open(fp, "rb"))):
            text = strip_text(ins["reviewText"])
            rating = int(float(ins["overall"]))
            review_id = ins["reviewerID"]
            x = {"business_id": ins["asin"],
                 "review_id": review_id,
                 "rating": rating,
                 "text": text}
            if spm is not None:
                piece = spm.Encode(text)
                if MIN_REV_LEN <= len(piece) <= MAX_REV_LEN:
                    x["piece"] = piece
                    d[ins["asin"]].append(x)
            else:
                d[ins["asin"]].append(x)
            p.update()

        for reviews in d.values():
            if len(reviews) > 10:
                yield from reviews
        obs.update(set(d))
    p.close()


class BertTokenizer:
    def __init__(self):
        from transformers import BertTokenizer
        self.tkzr = BertTokenizer.from_pretrained("bert-base-uncased")

    def Encode(self, s):
        return self.tkzr.encode(s)[1:-1]


def culpa(file_path: str, spm: SentencePieceProcessor = None):
    d = defaultdict(list)
    business_id = 0
    review_id = 0
    lns = []
    for ins in tqdm(map(json.loads, open(file_path)), desc="Culpa"):
        business_id += 1
        for text in ins['reviews']:
            review_id += 1
            text = strip_text(text)
            x = {"business_id": str(business_id),
                 "review_id": str(review_id),
                 # "rating": -1,
                 "text": text}
            piece = spm.Encode(text)
            # TODO: try larger reviews?
            lns.append(len(piece))
            if MIN_REV_LEN <= len(piece) <= MAX_REV_LEN:
                x["piece"] = piece
                d[business_id].append(x)

    print("Tokens stat:", file=sys.stderr)
    print("Avg: ", sum(lns) / len(lns), file=sys.stderr)
    print("Min:", min(lns), file=sys.stderr)
    print("Max:", max(lns), file=sys.stderr)
    for p in [5, 50, 75, 90, 95, 99]:
        print("  percentile %s:" % p, np.percentile(lns, p), file=sys.stderr)

    for reviews in d.values():
        if len(reviews) >= 4:
            yield from reviews


@click.command()
@click.argument("data_type", type=click.Choice(("yelp", "amzn", "culpa")), )
@click.argument("raw_file", type=click.Path(exists=True))
def main(data_type, raw_file):
    spm_file = Path(f"./data/sentencepiece/{data_type}4.model")
    if spm_file.exists():
        spm = SentencePieceProcessor()
        spm.Load(str(spm_file))
    else:
        spm = None

    if data_type == "yelp":
        parser = yelp
    elif data_type == "amzn":
        parser = amzn
    elif data_type == "culpa":
        parser = culpa
        #spm = BertTokenizer()
    else:
        raise KeyError()

    for x in parser(raw_file, spm):
        print(json.dumps(x))


if __name__ == '__main__':
    main()
