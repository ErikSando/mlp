#include <cassert>
#include <iostream>

#include "data/Dataset.hpp"

namespace mlp {
    Dataset::Dataset(const std::string& file_path) {
        m_file.open(file_path, std::ios::binary);

        if (!m_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

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
        m_totalLines = m_lineIndices.size();
    }

    void Dataset::readLine(const size_t line, char* buffer) {
        size_t start_line = line % m_totalLines;
        size_t next_line = (line + 1) % m_totalLines;

        std::streampos start = m_lineIndices[start_line], end;

        if (next_line - start_line == 1) end = m_lineIndices[next_line];
        else end = m_fileSize;

        size_t size = static_cast<size_t>(end - start);

        m_file.seekg(start);
        m_file.read(buffer, size);

        buffer[size] = '\0'; // temporary, to print correctly
    }

    size_t Dataset::parseBatch(const size_t start_line, const size_t batch_size, std::vector<InputLayer>& inputs) {
        // I think using reserve prevents repeated allocations without needing an assert
        // assert(inputs.size() == batch_size);

        inputs.reserve(batch_size);

        m_file.clear();

        char buffer[m_maxLineLength];

        // std::cout << m_maxLineLength << "\n\n";

        for (size_t i = 0; i < batch_size; i++) {
            readLine(start_line + i, buffer);

            // std::cout << buffer << "\n\n";

            inputs.emplace_back(784); // temporarily hard coded
            InputLayer& back = inputs.back();

            char* c = buffer;
            size_t data_index = 0;
            int value = 0;

            back.label = *c - '0'; // first character is the label
            c += 2; // move to the first non-label number

            while (*c != '\n' && c < buffer + m_maxLineLength) {
                if (*c == ',') {
                    back.data[data_index++] = value;
                    value = 0;
                }
                else if (std::isdigit(*c)) {
                    int digit = *c - '0';
                    value *= 10;
                    value += digit;
                }

                // ignore other characters

                c++; // say that again?
            }

            back.data[data_index++] = ((float) value) / 255.0f;
            value = 0;
        }

        return (start_line + batch_size) % m_totalLines;
    }
}