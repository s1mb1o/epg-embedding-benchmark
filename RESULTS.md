# Results

## st/all-MiniLM-L6-v2

```
'Evening News'          vs 'Evening News Broadcast'       -> 0.8930
'Morning Show'          vs 'Good Morning Program'         -> 0.6540
'Live Football'         vs 'Football Match Live'          -> 0.7774
'Documentary Premiere'  vs 'Documentary Film Premiere'    -> 0.9650
'Cooking Competition'   vs 'Culinary Contest'             -> 0.8236
mean: 0.8226
```

## openai/text-embedding-3-small

```
'Evening News'          vs 'Evening News Broadcast'       -> 0.8878
'Morning Show'          vs 'Good Morning Program'         -> 0.5742
'Live Football'         vs 'Football Match Live'          -> 0.7105
'Documentary Premiere'  vs 'Documentary Film Premiere'    -> 0.9425
'Cooking Competition'   vs 'Culinary Contest'             -> 0.7662
mean: 0.7762
```

Both models work — cosine similarity is high for close paraphrases (0.89–0.96) and lower for weaker synonyms (0.57–0.65). `all-MiniLM-L6-v2` edges out OpenAI on mean (0.82 vs 0.78).