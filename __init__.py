#!/usr/bin/env python

# Importing LSTM will import all dependencies
# from IdeaNets.models.lstm.scode import lstm as lstm
# import Synapsify
import os

# In the future as we add IdeaNets we will import IdeaNets.models.XXX

dirname = __path__[0]		# Package's main folder
__path__.insert(0, os.path.join(dirname, "IdeaNets"))

__all__ = ['models']
