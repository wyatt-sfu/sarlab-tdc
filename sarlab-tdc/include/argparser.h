/* Standard library headers */
#include <string_view>
#include <vector>

/**
 * Get a value for the specified option.
 */
std::string_view getOption(const std::vector<std::string_view> &args,
                           const std::string_view optName);

/**
 * Checks if an option exists
 */
bool hasOption(const std::vector<std::string_view> &args,
               const std::string_view &option_name);
