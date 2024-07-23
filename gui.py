import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QAction, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication
from sympy import symbols, Integral, pretty

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        TITLE = "Application of Advanced Optimization Methods and Algorithms in Nonlinear System Control"
        WINDOW_WIDTH = 800
        WINDOW_HEIGHT = 600
        
        self.setWindowTitle(TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.initUI()

    def initUI(self):
        # Suppress the default About and Preferences menu items on macOS
        QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar, True)
        
        # Create a menu bar
        menuBar = self.menuBar()
        
        # Create menu
        optionsMenu = menuBar.addMenu('Menu')
        
        # Create actions for menu
        mathModelAction = QAction('Mathematical/Simulation Model', self)
        mathModelAction.triggered.connect(self.showMathModel)
        
        controllerSynthesisAction = QAction('Controller Synthesis', self)
        controllerSynthesisAction.triggered.connect(self.showControllerSynthesis)
        
        simulationAction = QAction('Simulation', self)
        simulationAction.triggered.connect(self.showSimulation)
        
        aboutAction = QAction('About', self)
        aboutAction.triggered.connect(self.showAbout)
        
        # Add actions to menu
        optionsMenu.addAction(mathModelAction)
        optionsMenu.addAction(controllerSynthesisAction)
        optionsMenu.addAction(simulationAction)
        
        # Create a separate Help menu
        helpMenu = menuBar.addMenu('Help')
        helpMenu.addAction(aboutAction)
        
        # Initialize central widget area
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        
        # Layout for central widget
        self.layout = QVBoxLayout(self.centralWidget)
        
        # Placeholder widget for displaying content
        self.contentLabel = QLabel("Welcome to Application of Advanced Optimization Methods and Algorithms in Nonlinear System Control", self)
        self.layout.addWidget(self.contentLabel)

    def clearLayout(self):
        # Clear the layout by removing all widgets
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def showMathModel(self):
        self.clearLayout()
        x, a, b = symbols('x a b')
        integral_expr = Integral(x**2, (x, a, b))
        integral_text = pretty(integral_expr)
        self.renderMath(integral_text)

    def showControllerSynthesis(self):
        self.clearLayout()
        self.contentLabel = QLabel("Controller Synthesis content goes here.", self)
        self.layout.addWidget(self.contentLabel)

    def showSimulation(self):
        self.clearLayout()
        self.contentLabel = QLabel("Simulation content goes here.", self)
        self.layout.addWidget(self.contentLabel)

    def showAbout(self):
        self.clearLayout()
        about_text = (
            "<h2>About</h2>"
            "<p><strong>Application Name:</strong> Application of Advanced Optimization Methods and Algorithms in Nonlinear System Control</p>"
            "<p><strong>Version:</strong> 1.0.0</p>"
            "<p><strong>Author:</strong> Miroslav Uhlar</p>"
            "<p><strong>Email:</strong> uhlar.miro@gmai.com</p>"
            "<p><strong>GitHub:</strong> <a href='https://github.com/mirouhlar'>https://github.com/mirouhlar</a></p>"
            "<p>&copy; 2024 Miroslav Uhlar. All rights reserved.</p>"
        )
        self.contentLabel = QLabel(about_text, self)
        self.contentLabel.setTextFormat(Qt.RichText)
        self.layout.addWidget(self.contentLabel)
        
    def renderMath(self, math_text, position=None):
        # Create a new QLabel for the rendered math
        label = QLabel()
        label.setText(f"<pre>{math_text}</pre>")
        label.setAlignment(Qt.AlignCenter)
        
        # Add the QLabel to the layout
        if position is None:
            self.layout.addWidget(label)
        else:
            self.layout.insertWidget(position, label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
