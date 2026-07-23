#pragma once

#include <fstream>
#include <vector>

#include "nn/InputNodes.hpp"

/*
IDEAS

using this class to speed up file reading and parsing

construct line indices to quickly jump to lines and find line sizes

maybe use "mapping" in the future


*/

namespace mlp {
    constexpr size_t BUFFER_SIZE = 1 << 20; // 1 MB = up to 1 million chars (1 char = 1 byte), heap-allocated

    class Dataset {
        public:

        Dataset(const std::string& file_path);

        size_t parseBatch(const size_t start_line, const size_t batch_size, std::vector<InputNodes>& inputs);

        private:

        void generateLineIndices();
        void readLine(const size_t line, char* buffer);

        size_t m_maxLineLength = 0;
        size_t m_fileSize;
        size_t m_totalLines;
        std::ifstream m_file;
        std::vector<std::streampos> m_lineIndices;
    };
}