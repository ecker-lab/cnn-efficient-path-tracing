# Source Code for "CNNs efficiently learn long-range dependencies"

The implementation of narrow ResNet18 (nRN18) is provided in `models`. 
Our implementations of Pathfinder and cABC are provided in `datasets`. 
To download the data see https://openreview.net/forum?id=HJxrVA4FDS.

### Requirements

You need the python port of opencv (or replace loading and resizing with PIL) and numpy to run the code.
The datasets implement the PyTorch dataset API.

### Citation
```
@inproceedings{
lueddecke2020cnns,
    title={CNNs efficiently learn long-range dependencies},
    author={Timo L{\"u}ddecke and Alexander S Ecker},
    booktitle={NeurIPS 2020 Workshop SVRHM},
    year={2020},
    url={https://openreview.net/forum?id=dPwyQnHUVvw}
}
```
