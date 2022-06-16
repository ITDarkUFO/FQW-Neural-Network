import tkinter
import numpy
import scipy.special
import tkinter as tk
from tkinter import filedialog

class NeuralNetwork:
    # Инициализация нейронной сети
    def __init__(self, layers : int, nodes : tuple[int], learning_rate : float):
        # Количество слоев
        self.layers = layers
        
        # Количество узлов в слоях
        self.nodes = nodes
            
        # Коэффициент обучения
        self.lr = learning_rate
        
        # Рандомизация весов между узлами
        self.weights = [None] * (self.layers - 1)
        
        for i in range(self.layers - 1):
            self.weights[i] = numpy.random.normal(0.0, pow(self.nodes[i + 1], -0.5), (self.nodes[i + 1], self.nodes[i]))

        # Основное окно программы
        self.root = tk.Tk()
        self.root.withdraw()
        # self.root.iconbitmap()
        
    # Тренировка сети
    def train(self, inputs_list, targets_list):
        outputs = [None] * self.layers
        outputs[0] = numpy.array(inputs_list, ndmin=2).T
        
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Входные и выходные значения
        for i in range(1, self.layers):
            _inputs = numpy.dot(self.weights[i - 1], outputs[i - 1])
            outputs[i] = self.activation(_inputs)
        
        # Вычисление ошибки (В обратном порядке)
        errors = [None] * (self.layers - 1)
        errors[0] = targets - outputs[-1]
        
        for i in range(1, self.layers - 1):
            errors[i] = numpy.dot(self.weights[self.layers - 1 - i].T, errors[i - 1])
            
        errors = numpy.flip(errors, 0)

        # Обратное распространение ошибки
        for i in range(len(self.weights) - 1, -1, -1): # 1, 0
            self.weights[i] += self.lr * numpy.dot((errors[i] * outputs[i + 1] * (1.0 - outputs[i + 1])), numpy.transpose(outputs[i]))
    
    # Опрос сети
    def query(self, inputs_list):
        inputs = [None] * (self.layers - 1)
        outputs = [None] * self.layers
        
        outputs[0] = numpy.array(inputs_list, ndmin=2).T
        
        for i in range(1, len(outputs)):
            inputs[i - 1] = numpy.dot(self.weights[i - 1], outputs[i - 1])
            outputs[i] = self.activation(inputs[i - 1])
            
        return outputs[-1]
    
    # Функция активации (Сигмоида, позволяет как усиливать слабые сигналы, так и не насыщаться от сильных сигналов)
    def activation(self, x):
        return scipy.special.expit(x)
    
    # Экспортирование таблиц весов (Сохранение)
    def export_tables(self):
        file_path = filedialog.asksaveasfilename(
            filetypes = [("NPZ Archive", "*.npz"), ("All files", "*.*")],
            defaultextension=".npz",
            title = "Save file as",
            initialfile = str(self.layers) + "-ply NN weights data - " + str(self.nodes)
        )

        numpy.savez(file_path, numpy.array(self.weights, dtype=object))
    
    def get_weights(self):
        return self.weights
    
    # Импортирование таблиц весов (Загрузка)
    def import_tables(self):
        file_path = filedialog.askopenfilename(
            filetypes = [("NPZ Archive", "*.npz"), ("All files", "*.*")],
            title = "Open file",
        )
        
        data = numpy.load(file_path, allow_pickle=True)
        self.weights = data["arr_0"]