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

        // remove trailing newlines
        while (!m_lineIndices.empty() && m_lineIndices.back() == m_fileSize) {
            m_lineIndices.pop_back();
            m_fileSize--;
        }

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
    }

    // I want to clarify: I removed the first row of the MNIST data (column labels) to make this more simple
    // But I will probably make this class a little more flexible later so I can leave in the columns

    size_t Dataset::parseBatch(Batch& batch, const int start_line) {
        if (start_line != NO_LINE) m_currentLine = static_cast<size_t>(start_line);

        m_file.clear();

        std::vector<char> buffer(m_maxLineLength); // testing showed that making the buffer a member is slower

        size_t data_index = 0;

        for (size_t i = 0; i < batch.size; i++) {
            readLine(m_currentLine + i, buffer.data());

            char* c = buffer.data();
            int value = 0;

            batch.labels[i] = *c - '0'; // first character is the label
            c += 2; // move to the first non-label number

            while (c < buffer.data() + m_maxLineLength && *c != '\n') {
                if (*c == ',') {
                    batch.data[data_index++] = ((float) value) / 255.0f;
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

            batch.data[data_index++] = ((float) value) / 255.0f;
            value = 0;
        }

        m_currentLine = (m_currentLine + batch.size) % m_totalLines;

        return m_currentLine;
    }
}