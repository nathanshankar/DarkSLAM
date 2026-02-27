/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SPVOCABULARY_H
#define SPVOCABULARY_H

#include "Thirdparty/DBow3/src/DBoW3.h"   // or Vocabulary.h, whichever you are using
#include <string>

namespace ORB_SLAM3
{

class ORBVocabulary : public DBoW3::Vocabulary
{
public:
    // Keep the original interface used by ORB-SLAM3
    bool loadFromTextFile(const std::string &filename)
    {
        std::cout << "Loading vocabulary from file: " << filename << std::endl;
        this->load(filename);   // call the void-returning function
        std::cout << "Vocabulary loaded." << std::endl;

        // After loading, check if the vocabulary is valid.
        return !this->empty();
    }

    // (Optional) If ORB-SLAM3 ever calls saveToTextFile, you can map that too:
    // bool saveToTextFile(const std::string &filename) const
    // {
    //     return this->save(filename);
    // }
};

} // namespace ORB_SLAM3

#endif // SPVOCABULARY_H

