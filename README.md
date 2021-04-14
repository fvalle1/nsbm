[![DOI](https://zenodo.org/badge/DOI/10.TBA/TBA.svg)](https://doi.org/TBA)
[![Documentation Status](https://readthedocs.org/projects/trisbm/badge/?version=latest)](https://trisbm.readthedocs.io/en/latest/?badge=latest)


# tripartite Stochastic Block Modeling

Similar to [https://github.com/martingerlach/hSBM_Topicmodel](https://github.com/martingerlach/hSBM_Topicmodel) but with tripartite networks

The idea is to run SBM-based topic modeling on networks given keywords on documents

![network](network.png)

# Run

```bash
docker run -it -u jovyan -v $PWD:/home/jovyan/work -p 8899:8888 docker.pkg.github.com/fvalle1/trisbm/trisbm:latest
```

# Documentation

[Docs](https://fvalle1.github.io/trisbm/)

[Readthedocs](https://trisbm.readthedocs.io/en/latest/index.html)

# License

See [LICENSE](LICENSE).

This work [is in part based on](https://www.gnu.org/licenses/gpl-faq.en.html#WhyDoesTheGPLPermitUsersToPublishTheirModifiedVersions) [sbmtm](https://github.com/martingerlach/hSBM_Topicmodel)

## Third party libraries

This package depends on [graph-tool](https://graph-tool.skewed.de)
