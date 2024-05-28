This dataset is comprised of publicly available samples for the permutation of the following languages: "en", "da", "nb" and "sv".
The data is sourced from [Paracrawl](https://paracrawl.eu/), [CCAligned](https://opus.nlpl.eu/CCAligned/corpus/version/CCAligned) and [CCMatrix](https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix).

Altogether, there are 1854000 samples in this dataset split into train/test/valid buckets with proportions 80/10/10.

The representation for each language pair in each buckets is:
Train pairs:
```
{('sv', 'en'): 89513, ('sv', 'nb'): 152931, ('nb', 'da'): 156277, ('nb', 'sv'): 141412, ('da', 'en'): 103097, ('da', 'nb'): 150826, ('en', 'da'): 108729, ('en', 'nb'): 89786, ('en', 'sv'): 139502, ('sv', 'da'): 93665, ('da', 'sv'): 158201, ('nb', 'en'): 99447}
```
Valid pairs: 
```
{('nb', 'da'): 19328, ('sv', 'nb'): 18970, ('nb', 'en'): 12221, ('en', 'da'): 13656, ('sv', 'en'): 11295, ('en', 'nb'): 11063, ('nb', 'sv'): 17891, ('en', 'sv'): 17300, ('da', 'en'): 13007, ('da', 'nb'): 19091, ('da', 'sv'): 19937, ('sv', 'da'): 11448}
```

Test pairs: 
```
{('en', 'sv'): 17198, ('nb', 'da'): 19395, ('nb', 'en'): 12332, ('da', 'sv'): 19862, ('sv', 'nb'): 19099, ('nb', 'sv'): 17697, ('sv', 'en'): 11192, ('sv', 'da'): 11887, ('da', 'nb'): 19083, ('da', 'en'): 12896, ('en', 'da'): 13615, ('en', 'nb'): 11151}
```

The data is pre-cleaned for you.

For each bucket you have three files:
- *.src - the aligned source lines
- *.tgt - the aligned target lines
- *.metadata - the aligned metadata lines with info about the source and target language for each line