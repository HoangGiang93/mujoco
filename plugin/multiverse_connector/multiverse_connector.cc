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

#include <boost/algorithm/string/replace.hpp>
#ifdef __linux__
#include <jsoncpp/json/reader.h>
#elif _WIN32
#include <json/reader.h>
#endif

#include <mujoco/mujoco.h>

std::vector<int> get_sensor_ids(const mjModel *m, int instance)
{
  std::vector<int> sensor_ids;
  for (int sensor_id = 0; sensor_id < m->nsensor; sensor_id++)
  {
    if (m->sensor_type[sensor_id] == mjSENS_PLUGIN &&
        m->sensor_plugin[sensor_id] == instance)
    {
      sensor_ids.push_back(sensor_id);
    }
  }
  return sensor_ids;
}

namespace mujoco::plugin::multiverse_connector
{
  namespace
  {

    constexpr char server_host[] = "server_host";
    constexpr char server_port[] = "server_port";
    constexpr char client_port[] = "client_port";
    constexpr char world_name[] = "world_name";
    constexpr char simulation_name[] = "simulation_name";
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
    config.server_host = GetStringAttr(m, instance, server_host, config.server_host);
    config.server_port = GetStringAttr(m, instance, server_port, config.server_port);
    config.client_port = GetStringAttr(m, instance, client_port, config.client_port);
    config.world_name = GetStringAttr(m, instance, world_name, config.world_name);
    config.simulation_name = GetStringAttr(m, instance, simulation_name, config.simulation_name);

    const std::vector<int> sensor_ids = get_sensor_ids(m, instance);
    if (sensor_ids.empty())
    {
      mju_warning("Sensor not found for plugin instance %d", instance);
      return std::unique_ptr<MultiverseConnector>(new MultiverseConnector(config, m));
    }

    std::string send_str = GetStringAttr(m, instance, send);
    boost::replace_all(send_str, "'", "\"");
    Json::Value send_json;
    Json::Reader reader;
    if (reader.parse(send_str, send_json) && !send_str.empty())
    {
      for (const int sensor_id : sensor_ids)
      {
        if (m->sensor_objtype[sensor_id] == mjOBJ_BODY)
        {
          const int body_id = m->sensor_objid[sensor_id];
          const char *body_name = mj_id2name(m, mjtObj::mjOBJ_BODY, body_id);
          if (!body_name)
          {
            mju_warning_i("Body id %d must have a name\n", body_id);
          }
          else
          {
            config.send_objects[body_name] = {};

            for (const Json::Value &attribute_json : send_json[body_name])
            {
              const std::string attribute_name = attribute_json.asString();
              if (strcmp(attribute_name.c_str(), "position") == 0 ||
                  strcmp(attribute_name.c_str(), "quaternion") == 0)
              {
                config.send_objects[body_name].insert(attribute_name);
              }
            }
          }
        }
      }
    }
    else
    {
      mju_warning_s("Cannot parse %s into a map\n", send_str.c_str());
    }

    if (config.send_objects.empty())
    {
      mju_warning("No objects to send for plugin instance %d", instance);
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

    return std::unique_ptr<MultiverseConnector>(new MultiverseConnector(config, m));
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
    const std::vector<int> sensor_ids = get_sensor_ids(m, instance);
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

    // for (const int sensor_id : sensor_ids)
    // {
    //   const int body_id = m->sensor_objid[sensor_id];
    //   const char *body_name = mj_id2name(m, mjtObj::mjOBJ_BODY, body_id);
    //   printf("Time: %f - %d - %d - %s\n", d->time, instance, sensor_id, body_name);
    //   for (const std::string &attribute_name : config_.send_objects[body_name])
    //   {
    //     if (strcmp(attribute_name.c_str(), "position") == 0)
    //     {
    //       const mjtNum *position = d->xpos + 3 * body_id;
    //       printf("Position: %f - %f - %f\n", position[0], position[1], position[2]);
    //     }
    //     else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
    //     {
    //       const mjtNum *quaternion = d->xquat + 4 * body_id;
    //       printf("Quaternion: %f - %f - %f - %f\n", quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    //     }
    //   }
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

    std::vector<const char *> attributes = {server_host, server_port, client_port, world_name, simulation_name, send};
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

  MultiverseConnector::MultiverseConnector(MultiverseConfig config, const mjModel *m)
      : config_(std::move(config)), m_((mjModel *)m)
  {
    server_socket_addr = config_.server_host + ":" + config_.server_port;

    host = config_.server_host;
    port = config_.client_port;

    *world_time = 0.0;

    printf("Multiverse Server: %s - Multiverse Client: %s:%s\n", server_socket_addr.c_str(), host.c_str(), port.c_str());

    connect();
  }

  void MultiverseConnector::start_connect_to_server_thread()
  {
    connect_to_server();
  }

  void MultiverseConnector::wait_for_connect_to_server_thread_finish()
  {
  }

  void MultiverseConnector::start_meta_data_thread()
  {
    send_and_receive_meta_data();
  }

  void MultiverseConnector::wait_for_meta_data_thread_finish()
  {
  }

  bool MultiverseConnector::init_objects(bool from_request_meta_data)
  {
    if (from_request_meta_data)
    {
      if (request_meta_data_json["receive"].empty())
      {
        config_.receive_objects.clear();
      }
      if (request_meta_data_json["send"].empty())
      {
        config_.send_objects.clear();
      }
      for (const std::string &object_name : request_meta_data_json["receive"].getMemberNames())
      {
        for (const Json::Value &attribute_json : request_meta_data_json["receive"][object_name])
        {
          const std::string attribute_name = attribute_json.asString();
          config_.receive_objects[object_name].insert(attribute_name);
        }
      }
      for (const std::string &object_name : request_meta_data_json["send"].getMemberNames())
      {
        for (const Json::Value &attribute_json : request_meta_data_json["send"][object_name])
        {
          const std::string attribute_name = attribute_json.asString();
          config_.send_objects[object_name].insert(attribute_name);
        }
      }
    }

    std::set<std::string> objects_to_spawn;
    std::set<std::string> objects_to_destroy;
    for (const std::pair<const std::string, std::set<std::string>> &send_object : config_.send_objects)
    {
      const std::string object_name = send_object.first;
      if (strcmp(object_name.c_str(), "body") == 0 || strcmp(object_name.c_str(), "joint") == 0) // Skip if object name is "body" or "joint"
      {
        continue;
      }
      bool stop = !send_object.second.empty(); // Skip if object has no attributes
      for (const std::string &attribute_name : send_object.second)
      {
        if (strcmp(attribute_name.c_str(), "position") == 0 || strcmp(attribute_name.c_str(), "quaternion") == 0)
        {
          stop = false;
        }
      }
      if (stop)
      {
        continue;
      }

      if (mj_name2id(m_, mjtObj::mjOBJ_BODY, object_name.c_str()) == -1 &&
          mj_name2id(m_, mjtObj::mjOBJ_JOINT, object_name.c_str()) == -1 &&
          !(config_.receive_objects.count(object_name) > 0 &&
            config_.receive_objects[object_name].empty())) // If object does not exist as body or joint and has no receive attributes
      {
        objects_to_spawn.insert(object_name); // Then add object to spawn
      }
      else if (send_object.second.empty() == 0 &&
               config_.receive_objects.count(object_name) > 0 &&
               config_.receive_objects[object_name].empty()) // If object has no send attributes and has no receive attributes
      {
        objects_to_destroy.insert(object_name); // Then add object to destroy
      }
    }
    // for (const std::string &object_name : receive_objects_json.getMemberNames())
    // {
    //   if (strcmp(object_name.c_str(), "body") == 0 || strcmp(object_name.c_str(), "joint") == 0)
    //   {
    //     continue;
    //   }
    //   bool stop = true;
    //   for (const Json::Value &attribute_json : receive_objects_json[object_name])
    //   {
    //     const std::string attribute_name = attribute_json.asString();
    //     if (strcmp(attribute_name.c_str(), "position") == 0 || strcmp(attribute_name.c_str(), "quaternion") == 0)
    //     {
    //       stop = false;
    //     }
    //   }
    //   if (stop)
    //   {
    //     continue;
    //   }

    //   if (mj_name2id(m, mjtObj::mjOBJ_BODY, object_name.c_str()) == -1 &&
    //       mj_name2id(m, mjtObj::mjOBJ_JOINT, object_name.c_str()) == -1 &&
    //       mj_name2id(m, mjtObj::mjOBJ_ACTUATOR, object_name.c_str()) == -1 &&
    //       !(send_objects_json.isMember(object_name) &&
    //         send_objects_json[object_name].empty()))
    //   {
    //     objects_to_spawn.insert(object_name);
    //   }
    // }
    // for (const std::string &object_name : objects_to_destroy)
    // {
    //   receive_objects_json.removeMember(object_name);
    //   send_objects_json.removeMember(object_name);
    // }
    return true;
  }

  void MultiverseConnector::bind_request_meta_data()
  {
    const Json::Value api_callbacks = request_meta_data_json["api_callbacks"];
    const Json::Value api_callbacks_response = request_meta_data_json["api_callbacks_response"];
    request_meta_data_json.clear();

    if (!api_callbacks.isNull())
    {
      request_meta_data_json["api_callbacks"] = api_callbacks;
    }
    if (!api_callbacks_response.isNull())
    {
      request_meta_data_json["api_callbacks_response"] = api_callbacks_response;
    }

    request_meta_data_json["meta_data"]["world_name"] = config_.world_name;
    request_meta_data_json["meta_data"]["simulation_name"] = config_.simulation_name;
    request_meta_data_json["meta_data"]["length_unit"] = "m";
    request_meta_data_json["meta_data"]["angle_unit"] = "rad";
    request_meta_data_json["meta_data"]["mass_unit"] = "kg";
    request_meta_data_json["meta_data"]["time_unit"] = "s";
    request_meta_data_json["meta_data"]["handedness"] = "rhs";

    // for (const std::pair<std::string, std::set<std::string>> &send_object : config_.send_objects)
    // {
    //   const int body_id = mj_name2id(m, mjtObj::mjOBJ_BODY, send_object.first.c_str());
    //   const int joint_id = mj_name2id(m, mjtObj::mjOBJ_JOINT, send_object.first.c_str());
    //   const int actuator_id = mj_name2id(m, mjtObj::mjOBJ_ACTUATOR, send_object.first.c_str());
    //   if (body_id != -1)
    //   {
    //     const std::string body_name = send_object.first;
    //     for (const std::string &attribute_name : send_object.second)
    //     {
    //       request_meta_data_json["send"][body_name].append(attribute_name);
    //     }
    //   }
    //   else if (joint_id != -1)
    //   {
    //     const std::string joint_name = send_object.first;
    //     for (const std::string &attribute_name : send_object.second)
    //     {
    //       request_meta_data_json["send"][joint_name].append(attribute_name);
    //     }
    //   }
    //   else if (actuator_id != -1)
    //   {
    //     const std::string actuator_name = send_object.first;
    //     for (const std::string &attribute_name : send_object.second)
    //     {
    //       request_meta_data_json["send"][actuator_name].append(attribute_name);
    //     }
    //   }
    // }

    // for (const std::pair<std::string, std::set<std::string>> &receive_object : config_.receive_objects)
    // {
    //   const int body_id = mj_name2id(m, mjtObj::mjOBJ_BODY, receive_object.first.c_str());
    //   const int joint_id = mj_name2id(m, mjtObj::mjOBJ_JOINT, receive_object.first.c_str());
    //   const int actuator_id = mj_name2id(m, mjtObj::mjOBJ_ACTUATOR, receive_object.first.c_str());
    //   if (body_id != -1)
    //   {
    //     const std::string body_name = receive_object.first;
    //     for (const std::string &attribute_name : receive_object.second)
    //     {
    //       request_meta_data_json["receive"][body_name].append(attribute_name);
    //     }
    //   }
    //   else if (joint_id != -1)
    //   {
    //     const std::string joint_name = receive_object.first;
    //     const int qpos_id = m->jnt_qposadr[joint_id];
    //     for (const std::string &attribute_name : receive_object.second)
    //     {
    //       request_meta_data_json["receive"][joint_name].append(attribute_name);
    //     }
    //   }
    //   else if (actuator_id != -1)
    //   {
    //     const std::string actuator_name = receive_object.first;
    //     for (const std::string &attribute_name : receive_object.second)
    //     {
    //       request_meta_data_json["receive"][actuator_name].append(attribute_name);
    //     }
    //   }
    // }

    request_meta_data_str = request_meta_data_json.toStyledString();
  }

  void MultiverseConnector::bind_response_meta_data()
  {
    // bind_response_meta_data_callback();
  }

  void MultiverseConnector::bind_api_callbacks()
  {
    // if (!response_meta_data_dict.contains("api_callbacks"))
    // {
    //   return;
    // }
    // pybind11::list api_callbacks_list = response_meta_data_dict["api_callbacks"].cast<pybind11::list>();
    // for (size_t i = 0; i < pybind11::len(api_callbacks_list); i++)
    // {
    //   const pybind11::dict api_callback_dict = api_callbacks_list[i].cast<pybind11::dict>();
    //   for (auto api_callback_pair : api_callback_dict)
    //   {
    //     const std::string api_callback_name = api_callback_pair.first.cast<std::string>();
    //     if (api_callbacks.find(api_callback_name) == api_callbacks.end())
    //     {
    //       continue;
    //     }
    //     const pybind11::list api_callback_arguments = api_callback_pair.second.cast<pybind11::list>();
    //     api_callbacks[api_callback_name.c_str()](api_callback_arguments);
    //   }
    // }
  }

  void MultiverseConnector::bind_api_callbacks_response()
  {
    // if (!response_meta_data_dict.contains("api_callbacks"))
    // {
    //   return;
    // }
    // request_meta_data_dict["api_callbacks_response"] = pybind11::list();
    // pybind11::list api_callbacks_list = response_meta_data_dict["api_callbacks"].cast<pybind11::list>();
    // for (size_t i = 0; i < pybind11::len(api_callbacks_list); i++)
    // {
    //   const pybind11::dict api_callback_dict = api_callbacks_list[i].cast<pybind11::dict>();
    //   for (auto api_callback_pair : api_callback_dict)
    //   {
    //     const std::string api_callback_name = api_callback_pair.first.cast<std::string>();
    //     pybind11::dict api_callback_dict_request;
    //     if (api_callbacks_response.find(api_callback_name) != api_callbacks_response.end())
    //     {
    //       const pybind11::list api_callback_arguments = api_callback_pair.second.cast<pybind11::list>();
    //       api_callback_dict_request[api_callback_name.c_str()] = api_callbacks_response[api_callback_name.c_str()](api_callback_arguments);
    //     }
    //     else
    //     {
    //       api_callback_dict_request[api_callback_name.c_str()] = pybind11::list();
    //       api_callback_dict_request[api_callback_name.c_str()].cast<pybind11::list>().append("not implemented");
    //     }
    //     request_meta_data_dict["api_callbacks_response"].cast<pybind11::list>().append(api_callback_dict_request);
    //   }
    // }
  }

  void MultiverseConnector::clean_up()
  {
    // TODO: Find a clean way to clear the data because it's unsure if the data is still in use.

    // send_data.clear();

    // receive_data.clear();
  }

  void MultiverseConnector::reset()
  {
    // printf("[Client %s] Resetting the client (will be implemented).\n", port.c_str());
  }

  void MultiverseConnector::init_send_and_receive_data()
  {
    // if (send_buffer.buffer_double.size != send_data_double.size())
    // {
    //   send_data_double = std::vector<double>(send_buffer.buffer_double.size, 0.0);
    // }
    // if (send_buffer.buffer_uint8_t.size != send_data_uint8_t.size())
    // {
    //   send_data_uint8_t = std::vector<uint8_t>(send_buffer.buffer_uint8_t.size, 0);
    // }
    // if (send_buffer.buffer_uint16_t.size != send_data_uint16_t.size())
    // {
    //   send_data_uint16_t = std::vector<uint16_t>(send_buffer.buffer_uint16_t.size, 0);
    // }
    // if (receive_buffer.buffer_double.size != receive_data_double.size())
    // {
    //   receive_data_double = std::vector<double>(receive_buffer.buffer_double.size, 0.0);
    // }
    // if (receive_buffer.buffer_uint8_t.size != receive_data_uint8_t.size())
    // {
    //   receive_data_uint8_t = std::vector<uint8_t>(receive_buffer.buffer_uint8_t.size, 0);
    // }
    // if (receive_buffer.buffer_uint16_t.size != receive_data_uint16_t.size())
    // {
    //   receive_data_uint16_t = std::vector<uint16_t>(receive_buffer.buffer_uint16_t.size, 0);
    // }
  }

  void MultiverseConnector::bind_send_data()
  {
    // bind_send_data_callback();
    // if (send_data_double.size() != send_buffer.buffer_double.size || send_data_uint8_t.size() != send_buffer.buffer_uint8_t.size)
    // {
    //   printf("[Client %s] The size of in_send_data [%zu - %zu - %zu] does not match with send_buffer_size [%zu - %zu - %zu].\n",
    //          port.c_str(),
    //          send_data_double.size(),
    //          send_data_uint8_t.size(),
    //          send_data_uint16_t.size(),
    //          send_buffer.buffer_double.size,
    //          send_buffer.buffer_uint8_t.size,
    //          send_buffer.buffer_uint16_t.size);
    //   return;
    // }

    // std::copy(send_data_double.begin(), send_data_double.end(), send_buffer.buffer_double.data);
    // std::copy(send_data_uint8_t.begin(), send_data_uint8_t.end(), send_buffer.buffer_uint8_t.data);
    // std::copy(send_data_uint16_t.begin(), send_data_uint16_t.end(), send_buffer.buffer_uint16_t.data);
  }

  void MultiverseConnector::bind_receive_data()
  {
    // if (receive_data_double.size() != receive_buffer.buffer_double.size ||
    //     receive_data_uint8_t.size() != receive_buffer.buffer_uint8_t.size ||
    //     receive_data_uint16_t.size() != receive_buffer.buffer_uint16_t.size)
    // {
    //   printf("[Client %s] The size of receive_data [%zu - %zu - %zu] does not match with receive_buffer_size [%zu - %zu - %zu].\n",
    //          port.c_str(),
    //          receive_data_double.size(),
    //          receive_data_uint8_t.size(),
    //          receive_data_uint16_t.size(),
    //          receive_buffer.buffer_double.size,
    //          receive_buffer.buffer_uint8_t.size,
    //          receive_buffer.buffer_uint16_t.size);
    //   return;
    // }

    // std::copy(receive_buffer.buffer_double.data, receive_buffer.buffer_double.data + receive_buffer.buffer_double.size, receive_data_double.begin());
    // std::copy(receive_buffer.buffer_uint8_t.data, receive_buffer.buffer_uint8_t.data + receive_buffer.buffer_uint8_t.size, receive_data_uint8_t.begin());
    // std::copy(receive_buffer.buffer_uint16_t.data, receive_buffer.buffer_uint16_t.data + receive_buffer.buffer_uint16_t.size, receive_data_uint16_t.begin());
    // bind_receive_data_callback();
  }

} // namespace mujoco::plugin::multiverse_connector
