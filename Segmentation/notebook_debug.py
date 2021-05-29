# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
%load_ext autoreload
%autoreload 2

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from ProcessImage import ProcessImage
pimg = ProcessImage()
pimg.main()
from ProcessImage import busUI

import sys
import os
sys.executable
os.sys.path

def run():
    # pimg.load(np.arange(1,144)) #80, 101, 125
    # pimg.display7((80,))
    # pimg.runSaveGTStats(np.arange(1, 144))
    # pimg.segList.saveROIStats()
    # pimg.useBoxes()
    bus = busUI()
    bus.initUI()

run()




