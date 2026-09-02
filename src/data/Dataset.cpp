#include <cassert>
#include <iostream>

#include "data/Dataset.hpp"

namespace mlp {
    /*
        Scans numeric characters from the start to end addresses until hitting a non-numeric character.
        c - start address
        end - end address
        output - reference to the variable where the scanned value is written to
        If there is a non numeric character before the first numeric character, it will return an illegal character error.
        Any number of leading minus signs are permitted, each one negating the final value.
        Returns the error type, NONE if successful.
    */
    ScanError scanInt(char*& c, char* end, int& output) {
        int value = 0;
        bool is_negative = false;

        while (*c == '-' && c < end) {
            is_negative = !is_negative;
            c++;
        }

        if (c >= end) return ScanError::OUT_OF_BOUNDS;
        if (!std::isdigit(*c)) return ScanError::ILLEGAL_CHAR;

        while (std::isdigit(*c) && c < end) {
            int digit = *c - '0';
            value *= 10;
            value += digit;
            c++;
        }

        output = is_negative ? -value : value;

        return ScanError::NONE;
    }

    Dataset::Dataset(const std::string& file_path) {
        m_file.open(file_path, std::ios::binary);

        if (!m_file.is_open()) {
            return;
            // throw std::runtime_error("Failed to open file: " + file_path);
        }

        m_isSetup = true;

        generateLineIndices();
    }

    void Dataset::generateLineIndices() {
        std::vector<char> buffer(BUFFER_SIZE);
        std::streampos position = 0;

        m_lineIndices.push_back(0);

        while (m_file) {
            m_file.read(buffer.data(), BUFFER_SIZE);
            std::streamsize bytes_read = m_file.gcount();

            for (std::streamsize i = 0; i < bytes_read; i++) {
                if (buffer[i] == '\n') {
                    size_t line_index = static_cast<size_t>(position) + static_cast<size_t>(i) + 1;
                    size_t line_length = line_index - static_cast<size_t>(m_lineIndices.back());

                    if (line_length >= m_maxLineLength) m_maxLineLength = line_length + 1;

                    m_lineIndices.push_back(line_index);
                }
            }

            position += bytes_read;
        }

        m_fileSize = static_cast<size_t>(position);

        // remove trailing newlines
        while (!m_lineIndices.empty() && m_lineIndices.back() == m_fileSize) {
            m_lineIndices.pop_back();
            m_fileSize--;
        }

        m_totalLines = m_lineIndices.size();
    }

    size_t Dataset::readLine(const size_t line, char* buffer) {
        size_t start_line = line % m_totalLines;
        size_t next_line = (line + 1) % m_totalLines;

        std::streampos start = m_lineIndices[start_line], end;

        if (next_line - start_line == 1) end = m_lineIndices[next_line];
        else end = m_fileSize;

        size_t size = static_cast<size_t>(end - start);

        m_file.seekg(start);
        m_file.read(buffer, size);

        return size;
    }

    // I want to clarify: I removed the first row of the MNIST data (column labels) to make this more simple
    // But I will probably make this class a little more flexible later so I can leave in the columns

    size_t Dataset::parseBatch(Batch& batch, const int start_line) {
        if (start_line != NO_LINE) m_currentLine = static_cast<size_t>(start_line);

        m_file.clear();

        std::vector<char> buffer(m_maxLineLength); // testing showed that making the buffer a member is slower

        size_t data_index = 0;

        for (size_t i = 0; i < batch.size; i++) {
            size_t line_length = readLine(m_currentLine + i, buffer.data());

            char* c = buffer.data();
            int value = 0;

            char* end = buffer.data() + line_length;

            ScanError error = scanInt(c, end, batch.labels[i]);

            if (error != ScanError::NONE) {
                std::cerr << "Error while reading label: " << getErrorMsg(error) << "\n";
                throw std::runtime_error("Error while reading label");
            }

            while (c < end && *c != '\n' && *c != '\r') {
                if (*c != ',') {
                    std::cout << "'" << (int)(*c) << "'" << "\n";
                    std::cerr << "Invalid seperator character: '" << *c << "'\n";
                    throw std::runtime_error("");
                }

                c++;

                int value = 0;
                ScanError error = scanInt(c, end, value);

                if (error != ScanError::NONE) {
                    std::cerr << "Error while reading sample data: " << getErrorMsg(error) << "\n";
                    throw std::runtime_error("Error while reading sample data");
                }

                assert(data_index < batch.data.size());

                batch.data[data_index++] = ((float) value) / 255.0f;
            }
        }

        m_currentLine = (m_currentLine + batch.size) % m_totalLines;

        return m_currentLine;
    }
}