# CeleTrip

This is the source code for our paper.

## Data

## Prerequisites
The code has been successfully tested in the following environment. (For older PyG versions, you may need to modify the code)
- Python 3.8.12
- PyTorch 1.10.1
- Deep Graph Library 0.7.2
- Sklearn 1.0.1
- Pandas 1.3.5
- SpaCy 3.2.1
- Gensim 3.8.3

## Getting Started

### Prepare your data

We provide samples of our data in the `./Data` folder.

**Build Graph**

The file `build_trip_graph.py` is in `./preprocess/`
```python
python build_trip_graph.py
```

**Train model**
```python
python main.py
```


## Cite

Please cite our paper if you find this code useful for your research:

```
citation
```
