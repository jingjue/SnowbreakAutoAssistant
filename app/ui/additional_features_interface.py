# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'additional_features_interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_additional_features(object):
    def setupUi(self, additional_features):
        additional_features.setObjectName("additional_features")
        additional_features.resize(683, 667)
        self.gridLayout = QtWidgets.QGridLayout(additional_features)
        self.gridLayout.setObjectName("gridLayout")
        self.SegmentedWidget = SegmentedWidget(additional_features)
        self.SegmentedWidget.setObjectName("SegmentedWidget")
        self.gridLayout.addWidget(self.SegmentedWidget, 0, 0, 1, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(additional_features)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_fishing = QtWidgets.QWidget()
        self.page_fishing.setObjectName("page_fishing")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page_fishing)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.SimpleCardWidget = SimpleCardWidget(self.page_fishing)
        self.SimpleCardWidget.setObjectName("SimpleCardWidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.SimpleCardWidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.CheckBox_is_save_fish = CheckBox(self.SimpleCardWidget)
        self.CheckBox_is_save_fish.setObjectName("CheckBox_is_save_fish")
        self.gridLayout_5.addWidget(self.CheckBox_is_save_fish, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.BodyLabel = BodyLabel(self.SimpleCardWidget)
        self.BodyLabel.setObjectName("BodyLabel")
        self.horizontalLayout.addWidget(self.BodyLabel)
        self.SpinBox_fish_times = SpinBox(self.SimpleCardWidget)
        self.SpinBox_fish_times.setMinimum(1)
        self.SpinBox_fish_times.setMaximum(25)
        self.SpinBox_fish_times.setObjectName("SpinBox_fish_times")
        self.horizontalLayout.addWidget(self.SpinBox_fish_times)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.gridLayout_5.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem, 3, 0, 1, 1)
        self.BodyLabel_4 = BodyLabel(self.SimpleCardWidget)
        self.BodyLabel_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.BodyLabel_4.setWordWrap(True)
        self.BodyLabel_4.setObjectName("BodyLabel_4")
        self.gridLayout_5.addWidget(self.BodyLabel_4, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.SimpleCardWidget, 0, 0, 1, 1)
        self.PushButton_start_fishing = PushButton(self.page_fishing)
        self.PushButton_start_fishing.setObjectName("PushButton_start_fishing")
        self.gridLayout_2.addWidget(self.PushButton_start_fishing, 1, 0, 1, 1)
        self.textBrowser_log = QtWidgets.QTextBrowser(self.page_fishing)
        self.textBrowser_log.setStyleSheet("border-radius: 5px;\n"
"border: 2px;")
        self.textBrowser_log.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_log.setObjectName("textBrowser_log")
        self.gridLayout_2.addWidget(self.textBrowser_log, 0, 1, 2, 1)
        self.gridLayout_2.setColumnStretch(0, 2)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.stackedWidget.addWidget(self.page_fishing)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.BodyLabel_2 = BodyLabel(self.page_2)
        self.BodyLabel_2.setObjectName("BodyLabel_2")
        self.gridLayout_3.addWidget(self.BodyLabel_2, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.BodyLabel_3 = BodyLabel(self.page_3)
        self.BodyLabel_3.setObjectName("BodyLabel_3")
        self.gridLayout_4.addWidget(self.BodyLabel_3, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        self.gridLayout.addWidget(self.stackedWidget, 1, 0, 1, 1)

        self.retranslateUi(additional_features)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(additional_features)

    def retranslateUi(self, additional_features):
        _translate = QtCore.QCoreApplication.translate
        additional_features.setWindowTitle(_translate("additional_features", "Frame"))
        self.CheckBox_is_save_fish.setText(_translate("additional_features", "是否保存新纪录截图"))
        self.BodyLabel.setText(_translate("additional_features", "钓鱼次数："))
        self.BodyLabel_4.setText(_translate("additional_features", " * ﻿钓鱼每日有50次的上限,单个钓鱼点每天最多钓25次\n"
"* 当一个钓鱼点钓完后需要手动移动到下一个钓鱼点，进入钓鱼界面后再启动一次"))
        self.PushButton_start_fishing.setText(_translate("additional_features", "开始钓鱼"))
        self.BodyLabel_2.setText(_translate("additional_features", "待开发"))
        self.BodyLabel_3.setText(_translate("additional_features", "待开发"))
from qfluentwidgets import BodyLabel, CheckBox, PushButton, SegmentedWidget, SimpleCardWidget, SpinBox
