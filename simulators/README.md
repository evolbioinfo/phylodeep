# PhyloDeep tree simulators

This folder contains scripts used for tree simulation for PhyloDeep training.

Alternatively, you could install the python3 package [treesimulator](https://github.com/evolbioinfo/treesimulator), 
and simulate BD/BDEI/BDSS/MTBD trees with the following commands:

```bash
# Generate a BD tree with 200-500 tips, transmission rate of 0.5, 
# removal rate of 0.25 and sampling probability of 0.3
generate_bd --min_tips 200 --max_tips 500 --la 0.5 --psi 0.25 --p 0.3 --nwk tree.nwk --log params.csv

# Generate a BDEI tree with 200-500 tips, becoming-infectious rate of 1, transmission rate of 0.5, 
# removal rate of 0.25 and sampling probability of 0.3
generate_bdei --min_tips 200 --max_tips 500 --mu 1 --la 0.5 --psi 0.25 --p 0.3 --nwk tree.nwk --log params.csv

# Generate a BDSS tree with 200-500 tips, N-to-N transmission rate of 0.1, N-to-S transmission rate of 0.3, 
# S-to-N transmission rate of 0.5, S-to-S transmission rate of 1.5, 
# removal rate of 0.25 and sampling probability of 0.3
generate_bdss --min_tips 200 --max_tips 500 --la_nn 0.1 --la_ns 0.3 --la_sn 0.5 --la_ss 1.5 --psi 0.25 --p 0.3 --nwk tree.nwk --log params.csv
```

This package was used to generate very large trees in [Voznica _et al._ Nat Commun 2022](https://www.nature.com/articles/s41467-022-31511-0).


See [github.com/evolbioinfo/treesimulator](https://github.com/evolbioinfo/treesimulator) 
for more details on how to use/install the treesimulator package.

