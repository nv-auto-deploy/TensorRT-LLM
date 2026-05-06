/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/timestampUtils.h"

#include <cctype>
#include <cstddef>
#include <iostream>
#include <string>

namespace
{

bool isDigit(char const ch)
{
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

bool hasDateTimeSeparator(std::string const& timestamp)
{
    auto const separator = timestamp.find(' ');
    return separator != std::string::npos && separator > 0 && separator + 1 < timestamp.size();
}

bool endsWithSixDigits(std::string const& timestamp)
{
    constexpr std::size_t kMicrosecondDigits = 6;
    if (timestamp.size() < kMicrosecondDigits)
    {
        return false;
    }

    for (std::size_t index = timestamp.size() - kMicrosecondDigits; index < timestamp.size(); ++index)
    {
        if (!isDigit(timestamp[index]))
        {
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    std::string const timestamp = tensorrt_llm::common::getCurrentTimestamp();

    if (timestamp.empty())
    {
        std::cerr << "Timestamp is empty." << std::endl;
        return 1;
    }

    if (!hasDateTimeSeparator(timestamp))
    {
        std::cerr << "Timestamp lacks a date/time separator space: " << timestamp << std::endl;
        return 1;
    }

    if (!endsWithSixDigits(timestamp))
    {
        std::cerr << "Timestamp does not end with six digits: " << timestamp << std::endl;
        return 1;
    }

    return 0;
}
