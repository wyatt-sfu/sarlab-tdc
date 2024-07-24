#include <pybind11/pybind11.h>

namespace py = pybind11;

void hi_from_cpp()
{
    py::print("Hello !!!\n");
}

PYBIND11_MODULE(sarlabtdc, m)
{
    m.doc() = "python interface hello world";
    m.def("hello", &hi_from_cpp, "say hi!!!");
}
