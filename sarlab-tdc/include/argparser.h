/* Standard library headers */
#include <string>
#include <string_view>
#include <vector>

/**
 * Get a value for the specified option.
 */
std::string getArgValue(const std::vector<std::string_view> &args,
                        const std::string_view optName);

/**
 * Checks if an option exists
 */
bool hasArg(const std::vector<std::string_view> &args,
            const std::string_view &option_name);
