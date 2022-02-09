#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Perspecta Labs, Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading, time


mutex = threading.Lock()

class confusion_matrix(gr.sync_block, FigureCanvas):
    """
    docstring for block confusion_matrix
    """
    def __init__(self, class_labels):
        gr.sync_block.__init__(self,
            name="confusion_matrix",
            in_sig=[np.int16, np.int16],
            out_sig=None)
        
        self.labels = class_labels
        self.nclass = nclass = len(class_labels)
        self.tally = np.zeros((nclass,nclass))
        self.cm = np.zeros((len(class_labels),len(class_labels)))
        self.interpolation='nearest'
        self.cmap = plt.cm.Blues
        self.update_interval = 0.1
        self.backgroundColor = 'white'

        # self.fontColor = fontColor
        # self.backgroundColor = backgroundColor
        # self.ringColor = ringColor

        self.fig = Figure(figsize=(nclass, nclass))
        self.axes = self.fig.add_subplot(111)
        # self.fig.patch.set_facecolor(self.backgroundColor)
        # self.axes = self.fig.add_subplot(111, polar=True, facecolor=self.backgroundColor)

        FigureCanvas.__init__(self, self.fig)
        self.title = self.fig.suptitle('test', fontsize=8, fontweight='bold',
                                        color='black')

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.done = False
        threading.Thread(target=self.update_thread, daemon=True).start()

    def update_thread(self):
        while(not self.done):
            time.sleep(self.update_interval)

            with mutex: 
                # print(self.tally)
                for i in range(0,self.cm.shape[0]):
                    self.cm[i,:] = self.tally[i,:] / np.sum(self.tally[i,:])
                # print(self.cm)

            self.plot_confusion_matrix()

    def work(self, input_items, output_items):
        pred = input_items[0]
        actual = input_items[1]

        with mutex: 
            self.tally[actual, pred] = self.tally[actual, pred] + 1

        return len(input_items[0])

    def plot_confusion_matrix(self):
        self.axes.clear()
        self.axes.imshow(self.cm, interpolation=self.interpolation, cmap=self.cmap)
        # plt.title(title)
        # plt.colorbar()
        tick_marks = np.arange(len(self.labels))
        self.axes.set_xticks(tick_marks)
        self.axes.set_xticklabels(self.labels, rotation=45)
        self.axes.set_yticks(tick_marks)
        self.axes.set_yticklabels(self.labels, rotation=45)
        # plt.yticks(tick_marks, labels)
        # plt.tight_layout()

        # Loop over data dimensions and create text annotations.
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                self.axes.text(j, i, '{0:.2f}'.format(self.cm[i, j]),
                            ha="center", va="center", color="black", fontsize=6)

        self.axes.set_ylabel('True label')
        self.axes.set_xlabel('Predicted label')

        self.draw()

