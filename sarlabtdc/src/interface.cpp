/* Pybind headers */
#include <pybind11/pybind11.h>

/* Wrapper headers */
#include "tdcprocessor_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(sarlabtdc, m)
{
    m.doc() =
        "Time-domain correlation SAR processor for high bandwidth mmWave "
        "radars.";

    py::class_<TdcProcessorWrapper>(m, "TdcProcessor")
        .def(py::init<int>())
        .def("start", &TdcProcessorWrapper::start)
        .def("setRawData", &TdcProcessorWrapper::setRawData);
}
