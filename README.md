# technical_test_volta

Firstly, clone the repository on your computer.
  
Then, install the package with:

```python
pip install .
```

To launch the package, firstly download the data at https://physionet.org/content/ludb/1.0.1/
Then, use the following command:

```python
python test_volta/__main__.py PATH_TO_DATA
```

With `PATH_TO_DATA` being the path to the data folder of the LUDB database.

Then, after the program ended, a result folder has been created and contains 1 plot showing the label distribution in the dataset, and 3 confusion matrices, representing the result of the model calculations, with 3 types of normalization.
