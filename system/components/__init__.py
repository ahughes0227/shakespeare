"""The operators, one directory per family and one file per operator.

Every operator file holds the same things in the same order — NAME, FAMILY, OPERATION and
SUMMARY; the arguments it takes; what it produces; and how it runs — so opening one answers
what it is without reading a second file. The behaviour an operator calls lives in its
family's own logic modules, because behaviour is shared where operators are not: naming
serves both rendering and collision resolution.

The catalog, the argument models, the produced keys and the family dispatch tables are all
read from those files. A file is an operator when it declares its own NAME, which is how
`naming.py` and `name_render.py` sit side by side without ambiguity.
"""
