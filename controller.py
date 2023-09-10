from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
from DeepLearningTemplate.data_preparation import AudioLoader, parse_transforms
from tkinter import Button, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from playsound import playsound
import tkinter as tk
import gradio as gr
from src.predict_gui_model import Predict_gui_model
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")  # 使用 Qt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import *
import sys
from pathlib import Path
import os
from PyQt5.QtMultimedia import QSoundEffect, QMediaPlayer, QMediaContent

class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.wav'))
        self.predictor = Predict_gui_model(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        # self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.sample_rate = project_parameters.sample_rate
        assert project_parameters.threshold is not None, 'please check the threshold. the threshold value is {}.'.format(
            project_parameters.threshold)
        self.threshold = project_parameters.threshold
        self.web_interface = project_parameters.web_interface
        self.examples = project_parameters.examples if len(
            project_parameters.examples) else None

    def run(self, file):
        filename, result, threshold = self.predictor.Predict_gui_model(inputs=file)
        return filename, result, threshold

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.pushButton.clicked.connect(self.open_data)
        self.ui.pushButton_2.clicked.connect(self.predict)
        self.ui.pushButton_3.clicked.connect(self.time_domain)
        self.ui.pushButton_4.clicked.connect(self.frequency_domain)
        self.ui.pushButton_5.clicked.connect(self.mel_spectrogram)
        self.ui.pushButton_6.clicked.connect(self.play_audio)
        self.ui.pushButton_7.clicked.connect(self.stop_audio)

        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')          # 設定字體
        font.setPointSize(10)                      # 文字大小
        font.setBold(True)                         # 粗體

        self.ui.label.setFont(font)
        self.ui.label_3.setFont(font)
        self.ui.label_5.setFont(font)

#===============================================================================================

    def open_data(self):

        global fname
        filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
        fname = filePath

        file_path = str(fname)

        file_root = os.path.dirname(file_path)
        self.ui.label_2.setText(file_root)

        file_basename = os.path.basename(file_path)
        file_name = os.path.split(file_basename)[1]
        self.ui.label_4.setText(file_name)

    def predict(self):

        global fname
        file_path = str(fname)
        project_parameters = ProjectParameters().parse()
        if project_parameters.mode == 'predict_gui':
            filename, result, threshold = PredictGUI(project_parameters=project_parameters).run(file=file_path)

        font2 = QtGui.QFont()
        font2.setFamily('Microsoft YaHei')          # 設定字體
        font2.setPointSize(10)                      # 文字大小
        font2.setBold(True)                         # 粗體
        self.ui.label_6.setFont(font2)

        #normal
        if result < threshold: 
            self.ui.label_6.setText("該筆聲音預測為"+ "\t" +"<font color = #4682B4 , size = 7>"+"Normal"+"</font>")
        #abnormal
        else:
            self.ui.label_6.setText("該筆聲音預測為"+ "\t" + "<font color = #E60000 , size = 6>"+"Abnormal"+"</font>")

    def time_domain(self):

        global fname
        file_path = str(fname)
        waveform, sample_rate = torchaudio.load(filepath=file_path, normalize=False)
        waveform = waveform.float().mean(0)
        waveform = waveform.numpy()
        time = np.arange(0, waveform.size) * (1.0 / sample_rate)

        self.fig1 = plt.figure()

        self.ax1 = self.fig1.add_axes([0.17, 0.13, 0.7, 0.7])
        self.ax1.plot(time, waveform)
        self.ax1.set_title("time_domain")
        self.ax1.set_xlabel("time (seconds)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid()

        self.canvas1 = FigureCanvas(self.fig1)
        self.ui.gridLayout.addWidget(self.canvas1, 0, 0)

        self.canvas1.draw()

    def frequency_domain(self):

        global fname
        file_path = str(fname)
        waveform, sample_rate = torchaudio.load(filepath=file_path, normalize=False)
        waveform = waveform.float().mean(0)
        waveform = waveform.numpy()

        self.fig2 = plt.figure()

        fftdata = np.fft.rfft(waveform)
        freqs = np.linspace(0, sample_rate/2, fftdata.size)

        self.ax2 = self.fig2.add_axes([0.17, 0.13, 0.7, 0.7])
        self.ax2.plot(freqs, np.abs(fftdata), color='blue')
        self.ax2.set_title("frequency_domain")
        self.ax2.set_xlabel('Frequency(Hz)')
        self.ax2.set_ylabel('Amplitude')

        self.canvas2 = FigureCanvas(self.fig2)
        self.ui.gridLayout_2.addWidget(self.canvas2, 0, 0)

        self.canvas2.draw()

    def mel_spectrogram(self):

        global fname
        file_path = str(fname)
        waveform, sample_rate = torchaudio.load(filepath=file_path, normalize=False)
        waveform = waveform.float().mean(0)[None]

        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512 ,n_mels=64)(waveform)
        transform = torchaudio.transforms.AmplitudeToDB()
        specgram = transform(specgram)
        
        self.fig3 = plt.figure()

        self.ax3 = self.fig3.add_axes([0.2, 0.1, 0.7, 0.8])
        mel = self.ax3.imshow(specgram.detach()[0, :, :].numpy(), origin = 'lower')
        self.fig3.colorbar(mel, ax=self.ax3)
        self.ax3.set_title("mel spectrogram")
        self.ax3.set_xlabel('time')
        self.ax3.set_ylabel('Frequency(Hz)')   

        self.canvas3 = FigureCanvas(self.fig3)
        self.ui.gridLayout_4.addWidget(self.canvas3, 0, 0)

        self.canvas3.draw()

    def play_audio(self):

        global fname
        file_path = str(fname)
        # self.sound = QSound(file_path)
        # self.sound.play()
        self.player = QMediaPlayer()
        self.media_content = QMediaContent(QtCore.QUrl.fromLocalFile(file_path))
        self.player.setMedia(self.media_content)
        self.player.setVolume(80)
        self.player.play()

    def stop_audio(self):

        global fname
        file_path = str(fname)
        # self.sound = QSound(file_path)
        # self.sound.stop()
        self.player = QMediaPlayer()
        self.media_content = QMediaContent(QtCore.QUrl.fromLocalFile(file_path))
        self.player.setMedia(self.media_content)
        self.player.setVolume(80)
        self.player.stop()