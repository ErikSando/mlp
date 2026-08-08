#include <cassert>
#include <iostream>
#include <sstream>

#include "Commands.hpp"

namespace mlp {
    namespace commands {
        struct Colour {
            Colour(int r, int g, int b) : r(r), g(g), b(b) {}

            int r = 0, g = 0, b = 0;

            bool operator==(const Colour& other) const {
                return r == other.r && g == other.g && b == other.b;
            }

            Colour& operator=(const Colour& other) {
                if (this == &other) return *this;

                r = other.r;
                g = other.g;
                b = other.b;

                return *this;
            }
        };

        void set_colour(std::stringstream& ss, Colour& current, Colour& next) {
            if (current == next) return;

            assert(next.r >= 0);
            assert(next.r <= 255);
            assert(next.g >= 0);
            assert(next.g <= 255);
            assert(next.b >= 0);
            assert(next.b <= 255);

            ss << "\033[38;2;" << next.r << ";" << next.g << ";" << next.b << "m";

            current = next;
        }

        void print_sample(const ImageData& sample) {
            std::stringstream ss;
            
            Colour last_colour(0, 0, 0);
            ss << "\n\033[38;2;0;0;0m";

            int width = sample.width;
            int height = sample.height;

            for (int y = 0; y < sample.height; y++) {
                ss << "  ";

                for (int x = 0; x < sample.width; x++) {
                    float normal_value = sample.data.at(y * sample.width + x);
                    int value = (int) (normal_value * 255.0f);

                    Colour colour(value, value, value);
                    set_colour(ss, last_colour, colour);

                    ss << "\u2588\u2588";
                }

                ss << "\n";
            }

            ss << "\033[0m\n";

            std::cout << ss.str();
        }
    }
}