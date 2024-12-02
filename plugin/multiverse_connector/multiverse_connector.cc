// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "multiverse_connector.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <utility>
#include <boost/algorithm/string/replace.hpp>
#include <vector>
#ifdef __linux__
#include <jsoncpp/json/reader.h>
#elif _WIN32
#include <json/reader.h>
#endif

#include <mujoco/mujoco.h>

namespace mujoco::plugin::multiverse_connector
{
  namespace
  {

    constexpr char multiverse_server[] = "multiverse_server";
    constexpr char multiverse_client[] = "multiverse_client";
    constexpr char send[] = "send";

    std::string GetStringAttr(const mjModel *m, int instance, const char *attr, const std::string &default_value = "")
    {
      const char *value = mj_getPluginConfig(m, instance, attr);
      return (value != nullptr && value[0] != '\0') ? value : default_value;
    }

    // returns the next act given the current act_dot, after clamping, for native
    // mujoco dyntypes.
    // copied from engine_forward.
    // mjtNum NextActivation(const mjModel* m, const mjData* d, int actuator_id,
    //                       int act_adr, mjtNum act_dot) {
    //   mjtNum act = d->act[act_adr];

    //   if (m->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT) {
    //     // exact filter integration
    //     // act_dot(0) = (ctrl-act(0)) / tau
    //     // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
    //     //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
    //     mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id * mjNDYN]);
    //     act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
    //   } else {
    //     // Euler integration
    //     act = act + act_dot * m->opt.timestep;
    //   }

    //   // clamp to actrange
    //   if (m->actuator_actlimited[actuator_id]) {
    //     mjtNum* actrange = m->actuator_actrange + 2 * actuator_id;
    //     act = mju_clip(act, actrange[0], actrange[1]);
    //   }

    //   return act;
    // }

  } // namespace

  std::unique_ptr<MultiverseConnector> MultiverseConnector::Create(const mjModel *m, int instance)
  {
    MultiverseConfig config;
    config.server = GetStringAttr(m, instance, multiverse_server, config.server);
    config.client = GetStringAttr(m, instance, multiverse_client, config.client);

    printf("Multiverse Server: %s - Multiverse Client: %s\n", config.server.c_str(), config.client.c_str());
    for (int sensor_id = 0; sensor_id < m->nsensor; sensor_id++)
    {
      if (m->sensor_type[sensor_id] == mjSENS_PLUGIN &&
          m->sensor_plugin[sensor_id] == instance)
      {
        if (m->sensor_objtype[sensor_id] == mjOBJ_BODY)
        {
          const int body_id = m->sensor_objid[sensor_id];
          const char *body_name = mj_id2name(m, mjtObj::mjOBJ_BODY, body_id);
          if (!body_name)
          {
            mju_warning_i("Body id %d must have a name\n", body_id);
            continue;
          }
          std::string send_str = GetStringAttr(m, instance, send);
          boost::replace_all(send_str, "'", "\"");
          Json::Value send_json;
          Json::Reader reader;
          if (reader.parse(send_str, send_json) && !send_str.empty())
          {
            config.send_objects[body_name] = {};

            for (const Json::Value &attribute_json : send_json[body_name])
            {
              const std::string attribute_name = attribute_json.asString();
              if (strcmp(attribute_name.c_str(), "position") == 0 ||
                  strcmp(attribute_name.c_str(), "quaternion") == 0)
              {
                config.send_objects[body_name].push_back(attribute_name);
              }
            }
          }
          else
          {
            mju_warning_s("Cannot parse %s into a map\n", send_str.c_str());
          }
        }
      }
    }

    if (config.send_objects.empty())
    {
      mju_warning("send not found for plugin instance %d", instance);
    }

    // // Validate actnum values for all actuators:
    // for (int actuator_id : actuators) {
    //   int actnum = m->actuator_actnum[actuator_id];
    //   int expected_actnum = MultiverseConnector::ActDim(m, instance, actuator_id);
    //   int dyntype = m->actuator_dyntype[actuator_id];
    //   if (dyntype == mjDYN_FILTER || dyntype == mjDYN_FILTEREXACT ||
    //       dyntype == mjDYN_INTEGRATOR) {
    //     expected_actnum++;
    //   }
    //   if (actnum != expected_actnum) {
    //     mju_warning(
    //         "multiverse_connector %d has actdim %d, expected %d. Add actdim=\"%d\" to the "
    //         "multiverse_connector plugin element.",
    //         actuator_id, actnum, expected_actnum, expected_actnum);
    //     return nullptr;
    //   }
    // }
    return std::unique_ptr<MultiverseConnector>(new MultiverseConnector(config));
  }

  void MultiverseConnector::Reset(mjtNum *plugin_state) {}

  // mjtNum MultiverseConnector::GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
  //                     const State& state,
  //                     bool actearly) const {
  //   mjtNum ctrl = 0;
  //   if (m->actuator_dyntype[actuator_idx] == mjDYN_NONE) {
  //     ctrl = d->ctrl[actuator_idx];
  //     // clamp ctrl
  //     if (m->actuator_ctrllimited[actuator_idx]) {
  //       ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2 * actuator_idx],
  //                       m->actuator_ctrlrange[2 * actuator_idx + 1]);
  //     }
  //   } else {
  //     // Use of act instead of ctrl, to create integrated-velocity controllers or
  //     // to filter the controls.
  //     int actadr = m->actuator_actadr[actuator_idx] +
  //                  m->actuator_actnum[actuator_idx] - 1;
  //     if (actearly) {
  //       ctrl = NextActivation(m, d, actuator_idx, actadr, d->act_dot[actadr]);
  //     } else {
  //       ctrl = d->act[actadr];
  //     }
  //   }
  //   if (config_.slew_max.has_value() && state.previous_ctrl_exists) {
  //     mjtNum ctrl_min = state.previous_ctrl - *config_.slew_max * m->opt.timestep;
  //     mjtNum ctrl_max = state.previous_ctrl + *config_.slew_max * m->opt.timestep;
  //     ctrl = mju_clip(ctrl, ctrl_min, ctrl_max);
  //   }
  //   return ctrl;
  // }

  // void MultiverseConnector::ActDot(const mjModel* m, mjData* d, int instance) const {
  //   for (int actuator_idx : actuators_) {
  //     State state = GetState(m, d, actuator_idx);
  //     mjtNum ctrl = GetCtrl(m, d, actuator_idx, state, /*actearly=*/false);
  //     mjtNum error = ctrl - d->actuator_length[actuator_idx];

  //     int state_idx = m->actuator_actadr[actuator_idx];
  //     if (config_.i_gain) {
  //       mjtNum integral = state.integral + error * m->opt.timestep;
  //       if (config_.i_max.has_value()) {
  //         integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
  //       }
  //       d->act_dot[state_idx] = (integral - d->act[state_idx]) / m->opt.timestep;
  //       ++state_idx;
  //     }
  //     if (config_.slew_max.has_value()) {
  //       d->act_dot[state_idx] = (ctrl - d->act[state_idx]) / m->opt.timestep;
  //       ++state_idx;
  //     }
  //   }
  // }

  void MultiverseConnector::Compute(const mjModel *m, mjData *d, int instance)
  {
    // for (int i = 0; i < actuators_.size(); i++) {
    //   int actuator_idx = actuators_[i];
    //   State state = GetState(m, d, actuator_idx);
    //   mjtNum ctrl =
    //       GetCtrl(m, d, actuator_idx, state, m->actuator_actearly[actuator_idx]);

    //   mjtNum error = ctrl - d->actuator_length[actuator_idx];

    //   mjtNum ctrl_dot = m->actuator_dyntype[actuator_idx] == mjDYN_NONE
    //                         ? 0
    //                         : d->act_dot[m->actuator_actadr[actuator_idx] +
    //                                      m->actuator_actnum[actuator_idx] - 1];
    //   mjtNum error_dot = ctrl_dot - d->actuator_velocity[actuator_idx];

    //   mjtNum integral = 0;
    //   if (config_.i_gain) {
    //     integral = state.integral + error * m->opt.timestep;
    //     if (config_.i_max.has_value()) {
    //       integral =
    //           mju_clip(integral, -*config_.i_max, *config_.i_max);
    //     }
    //   }

    //   d->actuator_force[actuator_idx] = config_.p_gain * error +
    //                                     config_.d_gain * error_dot +
    //                                     config_.i_gain * integral;
    // }
  }

  void MultiverseConnector::Advance(const mjModel *m, mjData *d, int instance) const
  {
    // act variables already updated by MuJoCo integrating act_dot
  }

  int MultiverseConnector::StateSize(const mjModel *m, int instance)
  {
    return 0;
  }

  // int MultiverseConnector::ActDim(const mjModel* m, int instance, int actuator_id) {
  //   double i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
  //   return (i_gain ? 1 : 0) + (HasSlew(m, instance) ? 1 : 0);
  // }

  // MultiverseConnector::State MultiverseConnector::GetState(const mjModel* m, mjData* d, int actuator_idx) const {
  //   State state;
  //   int state_idx = m->actuator_actadr[actuator_idx];
  //   if (config_.i_gain) {
  //     state.integral = d->act[state_idx++];
  //   }
  //   if (config_.slew_max.has_value()) {
  //     state.previous_ctrl = d->act[state_idx++];
  //     state.previous_ctrl_exists = d->time > 0;
  //   }
  //   return state;
  // }

  void MultiverseConnector::RegisterPlugin()
  {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);
    plugin.name = "mujoco.multiverse_connector";
    plugin.capabilityflags |= mjPLUGIN_ACTUATOR | mjPLUGIN_SENSOR;

    std::vector<const char *> attributes = {multiverse_server, multiverse_client, send};
    plugin.nattribute = attributes.size();
    plugin.attributes = attributes.data();
    plugin.nstate = MultiverseConnector::StateSize;

    // Sensor dimension = number of all object attributes
    plugin.nsensordata = +[](const mjModel *m, int instance, int sensor_id)
    {
      std::string send_str = mj_getPluginConfig(m, instance, "send");
      boost::replace_all(send_str, "'", "\"");
      Json::Value send_json;
      Json::Reader reader;
      int nsensordata = 0;
      if (reader.parse(send_str, send_json) && !send_str.empty())
      {
        for (const std::string &object_name : send_json.getMemberNames())
        {
          nsensordata += send_json[object_name].size();
        }
      }
      return nsensordata;
    };

    plugin.init = +[](const mjModel *m, mjData *d, int instance)
    {
      std::unique_ptr<MultiverseConnector> multiverse_connector = MultiverseConnector::Create(m, instance);
      if (multiverse_connector == nullptr)
      {
        return -1;
      }
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(multiverse_connector.release());
      return 0;
    };
    plugin.destroy = +[](mjData *d, int instance)
    {
      delete reinterpret_cast<MultiverseConnector *>(d->plugin_data[instance]);
      d->plugin_data[instance] = 0;
    };
    plugin.reset = +[](const mjModel *m, mjtNum *plugin_state, void *plugin_data,
                       int instance)
    {
      auto *multiverse_connector = reinterpret_cast<MultiverseConnector *>(plugin_data);
      multiverse_connector->Reset(plugin_state);
    };
    plugin.compute =
        +[](const mjModel *m, mjData *d, int instance, int capability_bit)
    {
      auto *multiverse_connector = reinterpret_cast<MultiverseConnector *>(d->plugin_data[instance]);
      multiverse_connector->Compute(m, d, instance);
    };
    // plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    //   auto* pid = reinterpret_cast<MultiverseConnector*>(d->plugin_data[instance]);
    //   pid->Advance(m, d, instance);
    // };
    mjp_registerPlugin(&plugin);
  }

  MultiverseConnector::MultiverseConnector(MultiverseConfig config)
      : config_(std::move(config)) {}

} // namespace mujoco::plugin::multiverse_connector
