from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class ModelPlotCanvas(FigureCanvas):

    def __init__(self, rating_model, parent=None, width=5, height=4, dpi=100):

        # get an axes
        fig = Figure(figsize=(width, height), dpi=dpi)
        self._axes = fig.add_subplot(111)

        # add line for highlighted point
        hp_lines = self._axes.plot([None], [None], 'ko')
        self._highlighted_point = hp_lines[0]

        self._rating_model = rating_model
        self._observations_line = None
        self._point_selected = False
        self._plot_model()

        self._observation_numbers = self._get_observation_numbers()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.canvas.mpl_connect('pick_event', self._on_pick)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _get_observation_numbers(self):
        """

        :return:
        """

        line_data = self._observations_line.get_data(orig=True)

        model_data = self._rating_model.get_model_dataset()

        observation_numbers = np.arange(model_data.shape[0])+1

        x_variable = self._axes.get_xlabel()
        y_variable = self._axes.get_ylabel()

        x_index = np.in1d(model_data[x_variable], line_data[0])
        y_index = np.in1d(model_data[y_variable], line_data[1])

        return model_data.ix[x_index & y_index, 'Obs. number'].values

    def _on_click(self, event):
        """

        :param event:
        :return:
        """

        print('key=%s, button=%d, x=%d, y=%d, x_data=%f, y_data=%f' %
              (event.key, event.button, event.x, event.y, event.xdata, event.ydata))

        if self._point_selected:
            self._point_selected = False
        else:
            self._highlighted_point.set_marker('None')
            self._highlighted_point.figure.canvas.draw()
            # print('button=%d, x=%d, y=%d, x_data=%f, y_data=%f' %
            #       (event.button, event.x, event.y, event.xdata, event.ydata))
        # pass

    def _on_key(self, event):
        """

        :param event:
        :return:
        """
        print(event.key)

    def _on_pick(self, event):
        """

        :param event:
        :return:
        """

        this_line = event.artist

        if this_line is self._observations_line:
            x_data = this_line.get_xdata()
            y_data = this_line.get_ydata()

            ind = event.ind
            points = tuple(zip(self._observation_numbers[ind], x_data[ind], y_data[ind]))
            self._point_selected = True
            print('_on_pick points: ', points)

            # pressed_key = None
            # pressed_key = event.mouseevent.key

            # print(pressed_key, event.mouseevent.button)

            self._highlighted_point.set_data([x_data[ind]], [y_data[ind]])
            self._highlighted_point.set_marker('o')
            self._highlighted_point.figure.canvas.draw()

    def _plot_model(self):

        self._rating_model.plot(ax=self._axes)

        # find the line that shows the observations
        for line in self._axes.lines:
            if line.get_label() == 'Observations':
                self._observations_line = line
                break


class ModelPlotWindow(QtWidgets.QMainWindow):
    def __init__(self, rating_model):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        rm_plot_canvas = ModelPlotCanvas(rating_model)
        l.addWidget(rm_plot_canvas)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def file_quit(self):
        self.close()

    def closeEvent(self, ce):
        self.file_quit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About", "In development")
