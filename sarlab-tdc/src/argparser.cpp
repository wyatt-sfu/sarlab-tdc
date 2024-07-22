/* Standard library headers */
#include <string_view>
#include <vector>

/* Class header */
#include "argparser.h"

std::string_view getOption(const std::vector<std::string_view> &args,
                           const std::string_view optName)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == optName)
            if (it + 1 != end)
                return *(it + 1);
    }

    return "";
}

bool hasOption(const std::vector<std::string_view> &args,
               const std::string_view &optName)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == optName)
            return true;
    }

    return false;
}
