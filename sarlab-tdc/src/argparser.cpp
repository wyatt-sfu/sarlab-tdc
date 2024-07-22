/* Standard library headers */
#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

/* Class header */
#include "argparser.h"

std::string getArgValue(const std::vector<std::string_view> &args,
                        const std::string_view optName)
{
    std::string optVal;
    bool found = false;

    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == optName) {
            if (it + 1 != end) {
                optVal = *(it + 1);
                found = true;
                break;
            }
        }
    }

    if (!found) {
        return "";
    }

    // Trim whitespace from the start of the option value
    optVal.erase(optVal.begin(), std::find_if(optVal.begin(), optVal.end(),
                                              [](unsigned char ch) {
                                                  return !std::isspace(ch);
                                              }));

    // Trim whitespace from the end of the option value
    optVal.erase(
        std::find_if(optVal.rbegin(), optVal.rend(),
                     [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
        optVal.end());

    return optVal;
}

bool hasArg(const std::vector<std::string_view> &args,
            const std::string_view &optName)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == optName)
            return true;
    }

    return false;
}
