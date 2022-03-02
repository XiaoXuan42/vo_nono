#ifndef VO_NONO_CONFIG_H
#define VO_NONO_CONFIG_H

#include <string>

#include "vo_nono/frontend.h"
#include "vo_nono/system.h"

namespace vo_nono {
class Config
{
public:
    virtual SystemConfig generate_sysconf_from_str(const std::string &str) = 0;
};

class YamlConfig : public Config
{
public:
    SystemConfig generate_sysconf_from_str(const std::string &str) override;
    FrontendConfig generate_frontend_conf_from_str(const std::string &str);
};
}

#endif//VO_NONO_CONFIG_H
