#pragma once

namespace uammd{
namespace structured{
namespace Potentials{
namespace BasicParameters{

    namespace Pairs{

        struct LennardJones {

            struct InputPairParameters{

                std::string name_i;
                std::string name_j;

                real epsilon;
                real sigma;
            };

            struct PairParameters{
                real epsilon;
                real sigma;
            };

            template<typename T>
            static inline __host__ InputPairParameters readPairParameters(std::map<std::string,T>& info){

                InputPairParameters param;

                param.name_i = std::string(info.at("name_i"));
                param.name_j = std::string(info.at("name_j"));

                param.epsilon = real(info.at("epsilon"));
                param.sigma   = real(info.at("sigma"));

                return param;

            }

            static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){

                PairParameters param;

                param.epsilon = in_par.epsilon;
                param.sigma   = in_par.sigma;

                return param;
            }
        };
    }

}}}}