#pragma once

#include <iostream>
#include <source_location>
#include <string_view>

namespace mlp {
    namespace terminal_colours {
        constexpr char RED[] = "\033[31m";
        constexpr char YELLOW[] = "\033[33m";
        constexpr char DEFAULT[] = "\033[0m";
    }

    void error(std::string_view message, const std::source_location location = std::source_location::current());
    void warn(std::string_view message, const std::source_location location = std::source_location::current());
}