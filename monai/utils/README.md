
# Utilities

This contains various utility facilities and types. These are intended to be implemented in pure Python or using Numpy,
and not with Pytorch, so that these can be used outside of that context if need be. Some of the utilities implement 
core facilities:

* **aliases.py**: Implements the alias decorator and ability to define alias names for definitions which can be looked up.

* **moduleutils.py**: Implements the eager module loading facilities.

