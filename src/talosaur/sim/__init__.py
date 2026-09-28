"""Closed-loop simulator for guidance (numpy only): a 3-D world with animals in patches and depth
layers, simple vehicle dynamics driven by the real guidance commands, and a simulated camera that
produces the same outputs as the onboard model (frame logit, heatmap, patch tokens).

Used to compare search strategies and "how long to film" rules by encounters and footage per
hour before any sea time. It is a caricature of the ocean: its parameters come from the literature
(docs/SEARCH.md) and are meant to be replaced by what your own dives show.
"""
