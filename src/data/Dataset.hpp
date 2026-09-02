#pragma once

#include <fstream>
#include <vector>

#include "data/Batch.hpp"
#include "data/Sample.hpp"

/*

using this class to speed up file reading and parsing

IDEAS

(done) construct line indices to quickly jump to lines and find line sizes

maybe use "mapping" in the future

*/

namespace mlp {
    enum class ScanError {
        NONE, ILLEGAL_CHAR, OUT_OF_BOUNDS
    };

    inline std::string getErrorMsg(ScanError error) {
        switch (error) {
            case (ScanError::ILLEGAL_CHAR):
                return "illegal character";
            break;

            case (ScanError::OUT_OF_BOUNDS):
                return "no numeric characters within the given bounds";
            break;

            default:
                return "no error";
            break;
        }
    }

    constexpr size_t BUFFER_SIZE = 1 << 20; // 1 MB = up to 1 million chars (1 char = 1 byte), heap-allocated
    constexpr int NO_LINE = -1;

    class Dataset {
        public:

        Dataset(const std::string& file_path);

        size_t parseBatch(Batch& batch, const int start_line = NO_LINE);
        size_t size() { return m_totalLines; }

        void resetLine() { m_currentLine = 0; }

        bool isSetup() { return m_isSetup; }

        private:

        void generateLineIndices();

        size_t readLine(const size_t line, char* buffer);

        bool m_isSetup = false;

        size_t m_maxLineLength = 0;
        size_t m_fileSize;
        size_t m_totalLines;
        size_t m_currentLine = 0;

        std::ifstream m_file;
        std::vector<std::streampos> m_lineIndices;
    };
}