/* Pybind headers */
#include <pybind11/pybind11.h>

/* Wrapper headers */
#include "tdcprocessor_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(_sarlabtdc, m)
{
    m.doc() =
        "Time-domain correlation SAR processor for high bandwidth mmWave "
        "radars.";

    py::class_<TdcProcessorWrapper>(m, "TdcProcessor")
        .def(py::init<int>())
        .def("start", &TdcProcessorWrapper::start, py::arg("dopplerWinCenter"),
             py::arg("dopplerBandwidth"), py::arg("dopCentroidWin"),
             py::arg("applyRangeWin"), py::arg("dopplerWinTaper") = true)
        .def("setRawData", &TdcProcessorWrapper::setRawData)
        .def("setFocusGrid", &TdcProcessorWrapper::setFocusGrid)
        .def("getFocusedImage", &TdcProcessorWrapper::getFocusedImage);
}
