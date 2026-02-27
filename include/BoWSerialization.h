#ifndef DBOW3_SERIALIZATION_H
#define DBOW3_SERIALIZATION_H

#include "Thirdparty/DBow3/src/BowVector.h"
#include "Thirdparty/DBow3/src/FeatureVector.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
namespace serialization {

// ---- BowVector ----
template <class Archive>
void serialize(Archive & ar, DBoW3::BowVector & v, const unsigned int /*version*/)
{
    // BowVector in DBoW3 inherits from std::map<WordId, WordValue>
    ar & boost::serialization::base_object<
        std::map<DBoW3::WordId, DBoW3::WordValue>
    >(v);
}

// ---- FeatureVector (you'll likely hit this next) ----
template <class Archive>
void serialize(Archive & ar, DBoW3::FeatureVector & v, const unsigned int /*version*/)
{
    // FeatureVector typically inherits from std::map<NodeId, std::vector<unsigned int>>
    ar & boost::serialization::base_object<
        std::map<DBoW3::NodeId, std::vector<unsigned int>>
    >(v);
}

} // namespace serialization
} // namespace boost

#endif // DBOW3_SERIALIZATION_H
