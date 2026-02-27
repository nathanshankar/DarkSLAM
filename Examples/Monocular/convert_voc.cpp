// convert_voc.cpp
#include "Thirdparty/DBow3/src/Vocabulary.h"   // adjust if you use DBoW3.h instead
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " input_vocabulary.yml output_vocabulary.dbow3\n";
        return 1;
    }

    const std::string input  = argv[1];
    const std::string output = argv[2];

    std::cout << "Loading vocabulary from: " << input << std::endl;

    DBoW3::Vocabulary voc;
    voc.load(input);  // this is the same load() you use in ORB-SLAM3

    if (voc.empty())
    {
        std::cerr << "ERROR: loaded vocabulary is empty. Check the input file.\n";
        return 1;
    }

    std::cout << "Saving vocabulary to: " << output << std::endl;
    voc.save(output);   // this writes the fast binary .dbow3 format

    std::cout << "Done.\n";
    return 0;
}
