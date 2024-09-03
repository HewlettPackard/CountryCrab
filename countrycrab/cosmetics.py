import matplotlib.pyplot as plt
###
# Copyright (2024) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

from mpl_toolkits.axes_grid1 import Divider, Size

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (4, 4),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'figure.dpi':80,
         'figure.autolayout': False}
plt.rcParams.update(params)

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def new_fig(hs=3,vs=3):
    fig = plt.figure(figsize=(6, 6))
    
    # The first items are for padding and the second items are for the Axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(hs)]
    v = [Size.Fixed(0.7), Size.Fixed(vs)]
    
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))

    return ax
