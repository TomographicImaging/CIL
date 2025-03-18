
#include <nanobind/nanobind.h>
#include "axpby.h"
#include "Binning.h"
#include "FBP_filtering.h"
#include "FiniteDifferenceLibrary.h"

NB_MODULE(cilacc, m) {
	m.def("saxpby", &saxpby);
	m.def("daxpby", &daxpby);

	m.def("Binner_delete", &Binner_delete);
	m.def("Binner_new", &Binner_new);
	m.def("Binner_bin", &Binner_bin);

	m.def("filter_projections_avh", &filter_projections_avh);
	m.def("filter_projections_vah", &filter_projections_vah);

	m.def("openMPtest", &openMPtest);
	m.def("fdiff4D", &fdiff4D);
	m.def("fdiff3D", &fdiff3D);
	m.def("fdiff2D", &fdiff2D);

	m.doc() = "C-Extension for CIL";
}
