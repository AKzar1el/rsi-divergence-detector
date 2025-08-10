import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'RSI Divergence Detector'
copyright = '2025, Your Name'
author = 'Your Name'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
