/* Standard library headers */
#include <iostream>
#include <string_view>
#include <vector>

/**
 * =====================
 *      SARLAB-TDC
 * =====================
 *
 * This executable is used to run the SARlab time-domain correlation SAR
 * processor. This SAR processor is optimized to handle ultra high bandwidth
 * mmWave radars.
 *
 * Required Arguments:
 * -p <position filename>
 * -v <velocity filename>
 * -a <attitude filename>
 * -d <data filename>
 * -o <output filename>
 *
 * Optional Arguments:
 *
 */

int main(int argc, char *argv[])
{
    std::cout << "==============================\n";
    std::cout << "   SARLAB-TDC SAR PROCESSOR   \n";
    std::cout << "==============================\n";

    const std::vector<std::string_view> args(argv, argv + argc);

    return 0;
}
