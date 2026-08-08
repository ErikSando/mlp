// #include "io/Debug.hpp"

// #include <filesystem>

// namespace mlp {
//     void error(std::string_view message, const std::source_location location) {
//         std::cerr << terminal_colours::RED
//                   << "[Error]    " << terminal_colours::DEFAULT << message << "\n"
//                   << "File:      " << std::filesystem::path(location.file_name()).filename() << "\n"
//                   << "Line:      " << location.line() << "\n"
//                   << "Function:  " << location.function_name() << "\n";

//         std::abort();
//     }

//     void warn(std::string_view message, const std::source_location location) {
//         std::cerr << terminal_colours::YELLOW
//                   << "[Warning]  " << terminal_colours::DEFAULT << message << "\n"
//                   << "File:      " << std::filesystem::path(location.file_name()).filename() << "\n"
//                   << "Line:      " << location.line() << "\n"
//                   << "Function:  " << location.function_name() << "\n";
//     }
// }