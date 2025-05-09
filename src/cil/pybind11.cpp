
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "axpby.h"
#include "Binning.h"
#include "FBP_filtering.h"
#include "FiniteDifferenceLibrary.h"

namespace py = pybind11;

PYBIND11_MODULE(cilacc, m) {
	m.doc() = "C-Extension for CIL";

	m.def("saxpby", &saxpby, "float sa+by");
	m.def("daxpby", &daxpby, "double sa+by");

	m.def("Binner_delete", &Binner_delete, "Binner delete");
	m.def("Binner_new", &Binner_new, "Binner new");
	m.def("Binner_bin", &Binner_bin, "Binner bin");

	m.def("filter_projections_avh", &filter_projections_avh, "Filter Projections");
	m.def("filter_projections_vah", &filter_projections_vah, "Filter Projections");

	m.def("openMPtest", &openMPtest, "Open MP Test");
	m.def("fdiff4D", &fdiff4D, "fdiff4D desc");
	m.def("fdiff3D", &fdiff3D, "fdiff 3D desc");
	m.def("fdiff2D", &fdiff2D, "fdiff 2D desc");
}
