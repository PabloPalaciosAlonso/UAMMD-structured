#pragma once

#include <string>

#include "System/ExtendedSystem.cuh"

#include "Utils/Preprocessor/DataAccess.cuh"

#include "utils/Box.cuh"

namespace uammd{
namespace structured{
namespace Ensemble{

class EnsembleHandler{

    protected:

        std::string subType;

    public:

        EnsembleHandler(DataEntry& data){
            subType = data.getSubType();
        }

        std::string getSubType() const {
            return subType;
        }

        #define VARIABLE_IMPL(NAME, name, type) \
        virtual type get##NAME(){ \
            System::log<System::CRITICAL>("[Ensemble] %s not defined for ensemble \"%s\".", \
                                          std::string(#NAME).c_str(), subType.c_str()); \
            throw std::runtime_error("Variable not defined for ensemble."); \
        } \
        virtual void set##NAME(type value){ \
            System::log<System::CRITICAL>("[Ensemble] %s not defined for ensemble \"%s\".", \
                                          std::string(#NAME).c_str(), subType.c_str()); \
        }

        #define VARIABLE_AUX(NAME, name, type) VARIABLE_IMPL(NAME, name, type)

        #define VARIABLE(r, data, tuple) \
            VARIABLE_AUX(__DATA_CAPS__(tuple), __DATA_NAME__(tuple), __DATA_TYPE__(tuple))

        __MACRO_OVER_ENSEMBLE__(VARIABLE)

        #undef VARIABLE
        #undef VARIABLE_AUX
        #undef VARIABLE_IMPL

        virtual void updateDataEntry(DataEntry data) = 0;


};

}}}
