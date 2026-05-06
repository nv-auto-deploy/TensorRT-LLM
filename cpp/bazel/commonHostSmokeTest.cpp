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

#include "tensorrt_llm/common/stringUtils.h"

#include <iostream>
#include <string>

int main()
{
    auto const formatted = tensorrt_llm::common::fmtstr("%s-%d", "common", 7);
    if (formatted != "common-7")
    {
        std::cerr << "Unexpected formatted string: " << formatted << std::endl;
        return 1;
    }

    auto const values = tensorrt_llm::common::str2set("alpha,beta,alpha", ',');
    if (values.size() != 2 || values.count("alpha") != 1 || values.count("beta") != 1)
    {
        std::cerr << "Unexpected string set size: " << values.size() << std::endl;
        return 1;
    }

    return 0;
}
