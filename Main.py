import sys
import backend

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QScrollArea, QFrame
from PyQt6.QtGui import QPixmap

class Ventana(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Prueba de Modelo')
        self.resize(1000, 600)

        main_layout = QVBoxLayout()

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)

        scrollWidget = QWidget()
        scrollLayout = QVBoxLayout()

        # Sección de valores
        label_valores = QLabel('Valores del Modelo')
        label_valores.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_real = QLabel('Valor real de prueba.')
        self.label_predecido = QLabel('Valor predecido de prueba.')
        self.label_error = QLabel('Error relativo porcentual.')

        btn_prueba = QPushButton('Hacer Prueba', self)
        btn_prueba.clicked.connect(self.prueba)

        label_graficos = QLabel('Gráficos de entrenamiento')
        label_graficos.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label_accuracy = QLabel()
        label_accuracy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_accuracy.setPixmap(QPixmap('accuracy_prcp_prob_plot.png').scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        label_loss = QLabel()
        label_loss.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_loss.setPixmap(QPixmap('loss_plot.png').scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        label_mae = QLabel()
        label_mae.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_mae.setPixmap(QPixmap('mae_prcp_mm_plot.png').scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        scrollLayout.addWidget(label_valores)
        scrollLayout.addWidget(self.label_real)
        scrollLayout.addWidget(self.label_predecido)
        scrollLayout.addWidget(self.label_error)
        scrollLayout.addWidget(btn_prueba)
        scrollLayout.addWidget(label_graficos)
        scrollLayout.addWidget(label_accuracy)
        scrollLayout.addWidget(label_loss)
        scrollLayout.addWidget(label_mae)

        scrollWidget.setLayout(scrollLayout)
        self.scrollArea.setWidget(scrollWidget)
        main_layout.addWidget(self.scrollArea)
        self.setLayout(main_layout)

    def prueba(self):
        real, predict, error = backend.test()
        self.label_real.setText(f'Valor real de prueba: {real}')
        self.label_predecido.setText(f'Valor predecido de prueba: {predict}')
        self.label_error.setText(f'Error relativo porcentual: {error}%')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ventana()
    ex.show()
    sys.exit(app.exec())