#pragma once

namespace uammd{
namespace structured{

class VerletConditionalListSetFactory{

    public:

        using Creator = std::function<
        std::shared_ptr<VerletConditionalListSetBase>(
                std::shared_ptr<GlobalData>,
                std::shared_ptr<ParticleGroup>,
                DataEntry&,
                std::string)>;

        static VerletConditionalListSetFactory& getInstance() {
            static VerletConditionalListSetFactory instance;
            return instance;
        }

        void registerVerletConditionalListSet(const std::string& conditionType, Creator creator) {
            System::log<System::DEBUG>("[VerletConditionalListSetFactory] Registering VerletConditionalListSet in factory: %s",
                                         conditionType.c_str());
            if (getCreatorsRef().find(conditionType) != getCreatorsRef().end()) {
                System::log<System::CRITICAL>("[VerletConditionalListSetFactory] VerletConditionalListSet type already registered: %s",
                                              conditionType.c_str());
                throw std::runtime_error("VerletConditionalListSet type already registered");
            }
            getCreatorsRef()[conditionType] = creator;
        }

        std::shared_ptr<VerletConditionalListSetBase> createVerletConditionalListSet(const std::string& conditionType,
                                                                                     std::shared_ptr<GlobalData>    gd,
                                                                                     std::shared_ptr<ParticleGroup> pg,
                                                                                     DataEntry& data,
                                                                                     std::string name) {
            System::log<System::DEBUG>("[VerletConditionalListSetFactory] Creating VerletConditionalListSet: %s (type: %s)",
                                         name.c_str(), conditionType.c_str());

            auto& creators = getCreatorsRef();
            auto it = creators.find(conditionType);

            if (it != creators.end()) {
                return it->second(gd, pg, data, name);
            }
            System::log<System::CRITICAL>("[VerletConditionalListSetFactory] Unknown VerletConditionalListSet type: %s",
                                          conditionType.c_str());
            throw std::runtime_error("Unknown VerletConditionalListSet type");
        }

        const std::unordered_map<std::string, Creator>& getCreators() const {
            return getCreatorsRef();
        }

    private:

        VerletConditionalListSetFactory() = default;

        static std::unordered_map<std::string, Creator>& getCreatorsRef() {
            static std::unordered_map<std::string, Creator> creators;
            return creators;
        }
    };
}}

#define REGISTER_VERLET_CONDITIONAL_LIST_SET(type, ...) \
    namespace { \
        struct registerVCLS##type { \
            registerVCLS##type() { \
                if (__INCLUDE_LEVEL__ == 0) { \
                    uammd::structured::VerletConditionalListSetFactory::getInstance().registerVerletConditionalListSet( \
                        #type, [](std::shared_ptr<uammd::structured::GlobalData>    gd, \
                                  std::shared_ptr<uammd::ParticleGroup> pg, \
                                  uammd::structured::DataEntry& data, \
                                  std::string name) -> std::shared_ptr<uammd::structured::VerletConditionalListSetBase> { \
                        return std::make_shared<__VA_ARGS__>(gd, pg, data, name); \
                    }); \
                } \
            } \
        }; \
        registerVCLS##type registerVCLS##type##Instance; \
    }
