# Instructions to release an update to PiPy

1. Create distributables

```[python]
python -m build
```

2. Upload distributables

```[python]
python3 -m twine upload dist/*
```

Make sure that the ``dist`` directory only contains the distributables that
you want to push, or point to the files.

3. Release package also on GitHub.