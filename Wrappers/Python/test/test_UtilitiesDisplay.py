import unittest
import numpy as np
from cil.utilities import dataexample
from unittest.mock import patch
from utils import initialise_tests, has_matplotlib
initialise_tests()

if has_matplotlib:
    import matplotlib
    from cil.utilities.display import show1D


@unittest.skipUnless(has_matplotlib, "matplotlib not installed")
class TestShow1D(unittest.TestCase):

    def setUp(self):
        # Set up example data for testing
        self.data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        self.data2 = self.data * 0.8
        self.data3 = np.random.rand(10, 10, 10)


    @patch('matplotlib.pyplot.show')
    def test_show1D_figure_singledata(self, mock_show):
        # Test all components of the plot

        try:
            fig = show1D(self.data, slice_list=[(0, 64),(1,64)])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with all plot components: {e}")

        # Check the data
        line = fig.figure.axes[0].lines[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        self.assertEqual(len(x_data), 128)
        self.assertEqual(len(y_data), 128)
        np.testing.assert_array_equal(y_data, self.data.array[64, 64, :])

        # Check the plot components
        self.assertEqual(len(fig.figure.axes), 1)
        self.assertEqual(fig.figure.axes[0].get_title(), 'Slice at vertical:64, horizontal_y:64')
        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'horizontal_x index')
        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[0].get_xlim(), (0, 127))


    @patch('matplotlib.pyplot.show')
    def test_show1D_figure_multidata(self, mock_show):
        # Test all components of the plot

        try:
            fig = show1D([self.data, self.data2], slice_list=[(0, 64),(1,64)])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with all plot components: {e}")

        # Check the data
        line = fig.figure.axes[0].lines[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        self.assertEqual(len(x_data), 128)
        self.assertEqual(len(y_data), 128)
        np.testing.assert_array_equal(y_data, self.data.array[64, 64, :])

        line = fig.figure.axes[0].lines[1]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        self.assertEqual(len(x_data), 128)
        self.assertEqual(len(y_data), 128)
        np.testing.assert_array_equal(y_data, self.data2.array[64, 64, :])

        # Check the plot components
        self.assertEqual(len(fig.figure.axes), 1)
        self.assertEqual(fig.figure.axes[0].get_title(), 'Slice at vertical:64, horizontal_y:64')
        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'horizontal_x index')
        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[0].get_xlim(), (0, 127))
        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[0].get_text(), 'Dataset 0')
        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[1].get_text(), 'Dataset 1')


    @patch('matplotlib.pyplot.show')
    def test_show1D_figure_subplot(self, mock_show):
        # Test all components of the plot

        try:
            fig = show1D([self.data, self.data2], slice_list=None)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with all plot components: {e}")

        self.assertEqual(len(fig.figure.axes), 3)
        
        # Check the data
        line = fig.figure.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_ydata(), self.data.array[:, 64, 64])

        line = fig.figure.axes[0].lines[1]
        np.testing.assert_array_equal(line.get_ydata(), self.data2.array[:, 64, 64])

        line = fig.figure.axes[1].lines[0]
        np.testing.assert_array_equal(line.get_ydata(), self.data.array[64, :, 64])

        line = fig.figure.axes[1].lines[1]
        np.testing.assert_array_equal(line.get_ydata(), self.data2.array[64, :, 64])

        line = fig.figure.axes[2].lines[0]
        np.testing.assert_array_equal(line.get_ydata(), self.data.array[64, 64, :])

        line = fig.figure.axes[2].lines[1]
        np.testing.assert_array_equal(line.get_ydata(), self.data2.array[64, 64, :])


        self.assertEqual(fig.figure.axes[0].get_title(), 'Slice at horizontal_y:64, horizontal_x:64')
        self.assertEqual(fig.figure.axes[1].get_title(), 'Slice at vertical:64, horizontal_x:64')
        self.assertEqual(fig.figure.axes[2].get_title(), 'Slice at vertical:64, horizontal_y:64')

        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'vertical index')
        self.assertEqual(fig.figure.axes[1].get_xlabel(), 'horizontal_y index')
        self.assertEqual(fig.figure.axes[2].get_xlabel(), 'horizontal_x index')

        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[1].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[2].get_ylabel(), 'Value')

        self.assertEqual(fig.figure.axes[0].get_xlim(), (0, 127))
        self.assertEqual(fig.figure.axes[1].get_xlim(), (0, 127))
        self.assertEqual(fig.figure.axes[2].get_xlim(), (0, 127))

        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[0].get_text(), 'Dataset 0')
        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[1].get_text(), 'Dataset 1')

        self.assertEqual(fig.figure.axes[1].get_legend().get_texts()[0].get_text(), 'Dataset 0')
        self.assertEqual(fig.figure.axes[1].get_legend().get_texts()[1].get_text(), 'Dataset 1')

        self.assertEqual(fig.figure.axes[2].get_legend().get_texts()[0].get_text(), 'Dataset 0')
        self.assertEqual(fig.figure.axes[2].get_legend().get_texts()[1].get_text(), 'Dataset 1')


    @patch('matplotlib.pyplot.show')
    def test_show1D_inputs_slice_list(self, mock_show):
        # Test with slice_list as wrong length
        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[(1, 1)])
        self.assertEqual(str(cm.exception), "slice_list must provide a slice for ndim - 1 axes")

        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[(0, 1), (1, 1), (2, 1)])
        self.assertEqual(str(cm.exception), "slice_list must provide a slice for ndim - 1 axes")

        # Test with slice_list with duplicate axis
        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[(0, 1), ("Vertical", 2)])
        self.assertEqual(str(cm.exception), "slice_list contains duplicate axes. Each axis must be unique.")

        # Test with slice_list with invalid axis
        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[("Vertical", 1),("No", 1)])
        self.assertEqual(str(cm.exception), f"Invalid axis label: No")

        try:
            show1D(self.data, slice_list=None)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with single dataset and no slice_list: {e}")

        mock_show.reset_mock()

        try:
            show1D(self.data, slice_list=[("Horizontal_x", 30), ("Horizontal_y", 40)])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with single dataset and slice_list length 1: {e}")

        mock_show.reset_mock()

        try:
            show1D(self.data, slice_list=[(2, 30), (1, 40)])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with single dataset and slice_list length 1: {e}")

        mock_show.reset_mock()

        try:
            show1D(self.data, slice_list=[[(0, 62), (1, 70)], [(0, 61), (1, 70)], [(0, 67), (1, 70)]])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with single dataset and slice_list length 3: {e}")
   
        mock_show.reset_mock()

    @patch('matplotlib.pyplot.show')
    def test_show1D_inputs_datasets(self, mock_show):
        # Test with incompatible datasets
        with self.assertRaises(ValueError) as cm:
            show1D([self.data, self.data3])
        self.assertEqual(str(cm.exception), "All datasets must have the same shape")

        # Test with dataset_labels as wrong length
        with self.assertRaises(ValueError) as cm:
            show1D([self.data, self.data2], dataset_labels=["Data1"])
        self.assertEqual(str(cm.exception), "dataset_labels must be a list of strings equal to the number of datasets")

        with self.assertRaises(ValueError) as cm:
            show1D([self.data, self.data2], dataset_labels=["Data1", "Data2", "Data3"])
        self.assertEqual(str(cm.exception), "dataset_labels must be a list of strings equal to the number of datasets")


        try:
            fig = show1D([self.data, self.data2])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with default dataset_labels: {e}")

        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[0].get_text(), 'Dataset 0')
        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[1].get_text(), 'Dataset 1')
        mock_show.reset_mock()

        try:
            fig = show1D([self.data, self.data2], dataset_labels=["Data1", "Data2"])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed dataset_labels: {e}")

        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[0].get_text(), 'Data1')
        self.assertEqual(fig.figure.axes[0].get_legend().get_texts()[1].get_text(), 'Data2')
        mock_show.reset_mock()


    @patch('matplotlib.pyplot.show')
    def test_show1D_input_title(self, mock_show):

        # Test with title as wrong length
        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=None, title=["Title1"])
        self.assertEqual(str(cm.exception), "title must be a list of strings equal to the number of plots")

        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[(0, 30), (1, 40)], title=["Title1", "Title2"])
        self.assertEqual(str(cm.exception), "title must be a list of strings equal to the number of plots")


        try:
            fig = show1D(self.data, slice_list=None)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with default title: {e}")

        self.assertEqual(fig.figure.axes[0].get_title(), 'Slice at horizontal_y:64, horizontal_x:64')
        self.assertEqual(fig.figure.axes[1].get_title(), 'Slice at vertical:64, horizontal_x:64')
        self.assertEqual(fig.figure.axes[2].get_title(), 'Slice at vertical:64, horizontal_y:64')
        mock_show.reset_mock()

        try:
            fig = show1D(self.data, slice_list=None, title=["Title1", "Title2", "Title3"])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed title: {e}")

        self.assertEqual(fig.figure.axes[0].get_title(), 'Title1')
        self.assertEqual(fig.figure.axes[1].get_title(), 'Title2')
        self.assertEqual(fig.figure.axes[2].get_title(), 'Title3')


    @patch('matplotlib.pyplot.show')
    def test_show1D_input_axis_labels(self, mock_show):

        # Test with title as wrong length
        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=None, axis_labels=[("X", "Y")])
        self.assertEqual(str(cm.exception), "axis_labels must be a tuple or a list of tuples equal to the number of plots")

        with self.assertRaises(ValueError) as cm:
            show1D(self.data, slice_list=[(0, 30), (1, 40)], axis_labels=[("X", "Y"),("X", "Y"),("X", "Y")])
        self.assertEqual(str(cm.exception), "axis_labels must be a tuple or a list of tuples equal to the number of plots")

        try:
            fig = show1D(self.data, slice_list=None)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with default axis_labels: {e}")

        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'vertical index')
        self.assertEqual(fig.figure.axes[1].get_xlabel(), 'horizontal_y index')
        self.assertEqual(fig.figure.axes[2].get_xlabel(), 'horizontal_x index')

        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[1].get_ylabel(), 'Value')
        self.assertEqual(fig.figure.axes[2].get_ylabel(), 'Value')

        mock_show.reset_mock()

        try:
            fig = show1D(self.data, slice_list=None, axis_labels=("X", "Y"))
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed axis_labels tuple: {e}")

        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'X')
        self.assertEqual(fig.figure.axes[1].get_xlabel(), 'X')
        self.assertEqual(fig.figure.axes[2].get_xlabel(), 'X')
        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Y')
        self.assertEqual(fig.figure.axes[1].get_ylabel(), 'Y')
        self.assertEqual(fig.figure.axes[2].get_ylabel(), 'Y')

        mock_show.reset_mock()

        try:
            fig = show1D(self.data, slice_list=None, axis_labels=[("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3")])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed axis_labels list: {e}")

        self.assertEqual(fig.figure.axes[0].get_xlabel(), 'X1')
        self.assertEqual(fig.figure.axes[1].get_xlabel(), 'X2')
        self.assertEqual(fig.figure.axes[2].get_xlabel(), 'X3')

        self.assertEqual(fig.figure.axes[0].get_ylabel(), 'Y1')
        self.assertEqual(fig.figure.axes[1].get_ylabel(), 'Y2')
        self.assertEqual(fig.figure.axes[2].get_ylabel(), 'Y3')


    @patch('matplotlib.pyplot.show')
    def test_show1D_inputs_line_colours(self, mock_show):
        
        import logging
        logging.basicConfig(level=logging.WARNING)
        log = logging.getLogger("cil.utilities.display")
        # Test with line_colours as wrong length
        with self.assertLogs(log, level='WARNING') as cm:
            fig = show1D([self.data, self.data2], line_colours=["red"])
            mock_show.assert_called_once()

        self.assertIn('line_colours must be a list of colours at least as long as the number of datasets, using default colour palette', cm.output[0])

        # defaults '#377eb8', '#ff7f00'
        self.assertEqual(fig.figure.axes[0].lines[0].get_color(), '#377eb8')
        self.assertEqual(fig.figure.axes[0].lines[1].get_color(), '#ff7f00')

        mock_show.reset_mock()

        try:
            fig = show1D([self.data, self.data2], line_colours=["red", "blue"])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed line_colours: {e}")

        self.assertEqual(fig.figure.axes[0].lines[0].get_color(), 'red')
        self.assertEqual(fig.figure.axes[0].lines[1].get_color(), 'blue')


    @patch('matplotlib.pyplot.show')
    def test_show1D_inputs_line_styles(self, mock_show):
        import logging
        logging.basicConfig(level=logging.WARNING)
        log = logging.getLogger("cil.utilities.display")
        # Test with line_styles wrong length
        with self.assertLogs(log, level='WARNING') as cm:
            fig = show1D([self.data, self.data2], line_styles=["--"])
        self.assertIn('line_styles must be a list of styles at least as long as the number of datasets, using default line styles', cm.output[0])

        # defaults '-', '-.'
        self.assertEqual(fig.figure.axes[0].lines[0].get_linestyle(), '-')
        self.assertEqual(fig.figure.axes[0].lines[1].get_linestyle(), '--')

        mock_show.reset_mock()
        try:
            fig = show1D([self.data, self.data2], line_styles=[":", "-."])
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed line_styles: {e}")

        self.assertEqual(fig.figure.axes[0].lines[0].get_linestyle(), ':')
        self.assertEqual(fig.figure.axes[0].lines[1].get_linestyle(), '-.')


    @patch('matplotlib.pyplot.show')
    def test_show1D_figure_size(self, mock_show):

        try:
            fig = show1D(self.data, slice_list=None)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with default figsize: {e}")

        np.testing.assert_array_equal(fig.figure.get_size_inches(), (8,3*3))

        mock_show.reset_mock()
        try:
            fig = show1D(self.data, slice_list=None, size=(6, 6))
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show1D raised an exception with passed figsize: {e}")

        np.testing.assert_array_equal(fig.figure.get_size_inches(), (6,6*3))