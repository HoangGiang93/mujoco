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

int get_sensor_id(const mjModel *m, int obj_type, int obj_id)
{
  for (int sensor_id = 0; sensor_id < m->nsensor; sensor_id++)
  {
    if (m->sensor_objtype[sensor_id] == obj_type && m->sensor_objid[sensor_id] == obj_id)
    {
      return sensor_id;
    }
  }
  return -1;
}

Json::Value string_to_json(std::string &str)
{
  boost::replace_all(str, "'", "\"");
  if (str.empty())
  {
    mju_warning("Empty string cannot be converted to a map\n");
    return Json::Value();
  }
  Json::Value json;
  Json::Reader reader;
  if (reader.parse(str, json) && !str.empty())
  {
    return json;
  }
  else
  {
    mju_warning_s("Cannot parse %s into a map\n", str.c_str());
    return Json::Value();
  }
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
    //   mjtNum act = d_->act[act_adr];

    //   if (m_->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT) {
    //     // exact filter integration
    //     // act_dot(0) = (ctrl-act(0)) / tau
    //     // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
    //     //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
    //     mjtNum tau = mju_max(mjMINVAL, m_->actuator_dynprm[actuator_id * mjNDYN]);
    //     act = act + act_dot * tau * (1 - mju_exp(-m_->opt.timestep / tau));
    //   } else {
    //     // Euler integration
    //     act = act + act_dot * m_->opt.timestep;
    //   }

    //   // clamp to actrange
    //   if (m_->actuator_actlimited[actuator_id]) {
    //     mjtNum* actrange = m_->actuator_actrange + 2 * actuator_id;
    //     act = mju_clip(act, actrange[0], actrange[1]);
    //   }

    //   return act;
    // }

  } // namespace

  std::unique_ptr<MultiverseConnector> MultiverseConnector::Create(const mjModel *m, mjData *d, int instance)
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
      return std::unique_ptr<MultiverseConnector>(new MultiverseConnector(config, m, d));
    }

    std::string send_str = GetStringAttr(m, instance, send);
    boost::replace_all(send_str, "'", "\"");
    Json::Value send_json = string_to_json(send_str);
    Json::Reader reader;
    if (!send_json.empty())
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

    if (config.send_objects.empty())
    {
      mju_warning("No objects to send for plugin instance %d", instance);
    }

    // // Validate actnum values for all actuators:
    // for (int actuator_id : actuators) {
    //   int actnum = m_->actuator_actnum[actuator_id];
    //   int expected_actnum = MultiverseConnector::ActDim(m, instance, actuator_id);
    //   int dyntype = m_->actuator_dyntype[actuator_id];
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

    return std::unique_ptr<MultiverseConnector>(new MultiverseConnector(config, m, d));
  }

  void MultiverseConnector::Reset(mjtNum *plugin_state) {}

  // mjtNum MultiverseConnector::GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
  //                     const State& state,
  //                     bool actearly) const {
  //   mjtNum ctrl = 0;
  //   if (m_->actuator_dyntype[actuator_idx] == mjDYN_NONE) {
  //     ctrl = d_->ctrl[actuator_idx];
  //     // clamp ctrl
  //     if (m_->actuator_ctrllimited[actuator_idx]) {
  //       ctrl = mju_clip(ctrl, m_->actuator_ctrlrange[2 * actuator_idx],
  //                       m_->actuator_ctrlrange[2 * actuator_idx + 1]);
  //     }
  //   } else {
  //     // Use of act instead of ctrl, to create integrated-velocity controllers or
  //     // to filter the controls.
  //     int actadr = m_->actuator_actadr[actuator_idx] +
  //                  m_->actuator_actnum[actuator_idx] - 1;
  //     if (actearly) {
  //       ctrl = NextActivation(m, d, actuator_idx, actadr, d_->act_dot[actadr]);
  //     } else {
  //       ctrl = d_->act[actadr];
  //     }
  //   }
  //   if (config_.slew_max.has_value() && state.previous_ctrl_exists) {
  //     mjtNum ctrl_min = state.previous_ctrl - *config_.slew_max * m_->opt.timestep;
  //     mjtNum ctrl_max = state.previous_ctrl + *config_.slew_max * m_->opt.timestep;
  //     ctrl = mju_clip(ctrl, ctrl_min, ctrl_max);
  //   }
  //   return ctrl;
  // }

  // void MultiverseConnector::ActDot(const mjModel* m, mjData* d, int instance) const {
  //   for (int actuator_idx : actuators_) {
  //     State state = GetState(m, d, actuator_idx);
  //     mjtNum ctrl = GetCtrl(m, d, actuator_idx, state, /*actearly=*/false);
  //     mjtNum error = ctrl - d_->actuator_length[actuator_idx];

  //     int state_idx = m_->actuator_actadr[actuator_idx];
  //     if (config_.i_gain) {
  //       mjtNum integral = state.integral + error * m_->opt.timestep;
  //       if (config_.i_max.has_value()) {
  //         integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
  //       }
  //       d_->act_dot[state_idx] = (integral - d_->act[state_idx]) / m_->opt.timestep;
  //       ++state_idx;
  //     }
  //     if (config_.slew_max.has_value()) {
  //       d_->act_dot[state_idx] = (ctrl - d_->act[state_idx]) / m_->opt.timestep;
  //       ++state_idx;
  //     }
  //   }
  // }

  void MultiverseConnector::Compute(const mjModel *m, mjData *d, int instance)
  {
    // d_ = d;
    communicate();
    // printf("Time: %f\n", d_->time);
    const std::vector<int> sensor_ids = get_sensor_ids(m, instance);
    // for (int i = 0; i < actuators_.size(); i++) {
    //   int actuator_idx = actuators_[i];
    //   State state = GetState(m, d, actuator_idx);
    //   mjtNum ctrl =
    //       GetCtrl(m, d, actuator_idx, state, m_->actuator_actearly[actuator_idx]);

    //   mjtNum error = ctrl - d_->actuator_length[actuator_idx];

    //   mjtNum ctrl_dot = m_->actuator_dyntype[actuator_idx] == mjDYN_NONE
    //                         ? 0
    //                         : d_->act_dot[m_->actuator_actadr[actuator_idx] +
    //                                      m_->actuator_actnum[actuator_idx] - 1];
    //   mjtNum error_dot = ctrl_dot - d_->actuator_velocity[actuator_idx];

    //   mjtNum integral = 0;
    //   if (config_.i_gain) {
    //     integral = state.integral + error * m_->opt.timestep;
    //     if (config_.i_max.has_value()) {
    //       integral =
    //           mju_clip(integral, -*config_.i_max, *config_.i_max);
    //     }
    //   }

    //   d_->actuator_force[actuator_idx] = config_.p_gain * error +
    //                                     config_.d_gain * error_dot +
    //                                     config_.i_gain * integral;
    // }

    // for (const int sensor_id : sensor_ids)
    // {
    //   const int body_id = m_->sensor_objid[sensor_id];
    //   const char *body_name = mj_id2name(m, mjtObj::mjOBJ_BODY, body_id);
    //   printf("Time: %f - %d - %d - %s\n", d_->time, instance, sensor_id, body_name);
    //   for (const std::string &attribute_name : config_.send_objects[body_name])
    //   {
    //     if (strcmp(attribute_name.c_str(), "position") == 0)
    //     {
    //       const mjtNum *position = d_->xpos + 3 * body_id;
    //       printf("Position: %f - %f - %f\n", position[0], position[1], position[2]);
    //     }
    //     else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
    //     {
    //       const mjtNum *quaternion = d_->xquat + 4 * body_id;
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
  //   int state_idx = m_->actuator_actadr[actuator_idx];
  //   if (config_.i_gain) {
  //     state.integral = d_->act[state_idx++];
  //   }
  //   if (config_.slew_max.has_value()) {
  //     state.previous_ctrl = d_->act[state_idx++];
  //     state.previous_ctrl_exists = d_->time > 0;
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
    plugin.nsensordata = +[](const mjModel *m, int instance, int sensor_id)
    {
      std::string send_str = GetStringAttr(m, instance, send);
      const Json::Value send_json = string_to_json(send_str);
      if (send_json.empty())
      {
        mju_warning("Send list is empty\n");
        return 0;
      }
      if (m->sensor_objtype[sensor_id] == mjOBJ_UNKNOWN)
      {
        return 10;
      }
      else if (m->sensor_objtype[sensor_id] == mjOBJ_BODY)
      {
        const int body_id = m->sensor_objid[sensor_id];
        const char *body_name = mj_id2name(m, mjtObj::mjOBJ_BODY, body_id);
        if (!body_name)
        {
          mju_warning_i("Body id %d must have a name\n", body_id);
          return 0;
        }
        if (!send_json.isMember(body_name))
        {
          mju_warning_s("Body %s is not in the send list\n", body_name);
          return 0;
        }

        if (send_json[body_name].empty())
        {
          mju_warning_s("Body %s has no attributes in the send list\n", body_name);
          return 0;
        }
        int nsensordata = 0;
        for (const Json::Value &attribute_json : send_json[body_name])
        {
          const std::string attribute_name = attribute_json.asString();
          if (strcmp(attribute_name.c_str(), "position") == 0)
          {
            nsensordata += 3;
          }
          else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
          {
            nsensordata += 4;
          }
        }
        return nsensordata;
      }
      else
      {
        mju_warning("Sensor object type %d is not supported\n", m->sensor_objtype[sensor_id]);
        return 0;
      }
    };

    plugin.init = +[](const mjModel *m, mjData *d, int instance)
    {
      std::unique_ptr<MultiverseConnector> multiverse_connector = MultiverseConnector::Create(m, d, instance);
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
    //   auto* pid = reinterpret_cast<MultiverseConnector*>(d_->plugin_data[instance]);
    //   pid_->Advance(m, d, instance);
    // };
    mjp_registerPlugin(&plugin);
  }

  MultiverseConnector::MultiverseConnector(MultiverseConfig config, const mjModel *m, mjData *d)
      : config_(std::move(config)), m_((mjModel *)m), d_(d)
  {
    server_socket_addr = config_.server_host + ":" + config_.server_port;

    host = config_.server_host;
    port = config_.client_port;

    *world_time = 0.0;

    printf("Multiverse Server: %s - Multiverse Client: %s:%s\n", server_socket_addr.c_str(), host.c_str(), port.c_str());

    connect();

    communicate(true);
  }

  void MultiverseConnector::start_connect_to_server_thread()
  {
    printf("Start connect_to_server thread\n");
    connect_to_server();
    printf("End connect_to_server thread\n");
  }

  void MultiverseConnector::wait_for_connect_to_server_thread_finish()
  {
  }

  void MultiverseConnector::start_meta_data_thread()
  {
    printf("Start send_and_receive_meta_data thread\n");
    send_and_receive_meta_data();
    printf("End send_and_receive_meta_data thread\n");
  }

  void MultiverseConnector::wait_for_meta_data_thread_finish()
  {
  }

  bool MultiverseConnector::init_objects(bool from_request_meta_data)
  {
    printf("Init objects\n");
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
        printf("Object name: %s Attribute name: %s\n", object_name.c_str(), attribute_name.c_str());
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
          mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, object_name.c_str()) == -1 &&
          !(config_.receive_objects.count(object_name) > 0 &&
            config_.receive_objects[object_name].empty())) // If object does not exist as body or joint or actuator and has no receive attributes
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

    //   if (mj_name2id(m_, mjtObj::mjOBJ_BODY, object_name.c_str()) == -1 &&
    //       mj_name2id(m_, mjtObj::mjOBJ_JOINT, object_name.c_str()) == -1 &&
    //       mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, object_name.c_str()) == -1 &&
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
    printf("End Init objects\n");
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

    for (const std::pair<const std::string, std::set<std::string>> &send_object : config_.send_objects)
    {
      const int body_id = mj_name2id(m_, mjtObj::mjOBJ_BODY, send_object.first.c_str());
      const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, send_object.first.c_str());
      const int actuator_id = mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, send_object.first.c_str());
      if (body_id != -1)
      {
        const std::string body_name = send_object.first;
        for (const std::string &attribute_name : send_object.second)
        {
          request_meta_data_json["send"][body_name].append(attribute_name);
        }
      }
      if (joint_id != -1)
      {
        const std::string joint_name = send_object.first;
        for (const std::string &attribute_name : send_object.second)
        {
          request_meta_data_json["send"][joint_name].append(attribute_name);
        }
      }
      if (actuator_id != -1)
      {
        const std::string actuator_name = send_object.first;
        for (const std::string &attribute_name : send_object.second)
        {
          request_meta_data_json["send"][actuator_name].append(attribute_name);
        }
      }
    }

    for (const std::pair<const std::string, std::set<std::string>> &receive_object : config_.receive_objects)
    {
      const int body_id = mj_name2id(m_, mjtObj::mjOBJ_BODY, receive_object.first.c_str());
      const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, receive_object.first.c_str());
      const int actuator_id = mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, receive_object.first.c_str());
      if (body_id != -1)
      {
        const std::string body_name = receive_object.first;
        for (const std::string &attribute_name : receive_object.second)
        {
          request_meta_data_json["receive"][body_name].append(attribute_name);
        }
      }
      if (joint_id != -1)
      {
        const std::string joint_name = receive_object.first;
        for (const std::string &attribute_name : receive_object.second)
        {
          request_meta_data_json["receive"][joint_name].append(attribute_name);
        }
      }
      if (actuator_id != -1)
      {
        const std::string actuator_name = receive_object.first;
        for (const std::string &attribute_name : receive_object.second)
        {
          request_meta_data_json["receive"][actuator_name].append(attribute_name);
        }
      }
    }

    request_meta_data_str = request_meta_data_json.toStyledString();
  }

  void MultiverseConnector::bind_response_meta_data()
  {
    for (const std::pair<const std::string, std::set<std::string>> &send_object : config_.send_objects)
    {
      const int body_id = mj_name2id(m_, mjtObj::mjOBJ_BODY, send_object.first.c_str());
      const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, send_object.first.c_str());
      const int mocap_id = m_->body_mocapid[body_id];
      const int actuator_id = mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, send_object.first.c_str());
      if (body_id != -1)
      {
        if (mocap_id != -1)
        {
          for (const std::string &attribute_name : send_object.second)
          {
            const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              const Json::Value x_json = attribute_data[0];
              const Json::Value y_json = attribute_data[1];
              const Json::Value z_json = attribute_data[2];
              if (!x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                d_->mocap_pos[3 * mocap_id] = x_json.asDouble();
                d_->mocap_pos[3 * mocap_id + 1] = y_json.asDouble();
                d_->mocap_pos[3 * mocap_id + 2] = z_json.asDouble();
              }
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              const Json::Value w_json = attribute_data[0];
              const Json::Value x_json = attribute_data[1];
              const Json::Value y_json = attribute_data[2];
              const Json::Value z_json = attribute_data[3];
              if (!w_json.isNull() && !x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                d_->mocap_quat[4 * mocap_id] = w_json.asDouble();
                d_->mocap_quat[4 * mocap_id + 1] = x_json.asDouble();
                d_->mocap_quat[4 * mocap_id + 2] = y_json.asDouble();
                d_->mocap_quat[4 * mocap_id + 3] = z_json.asDouble();
              }
            }
            else
            {
              mju_warning("Attribute %s not supported for body %s\n", attribute_name.c_str(), send_object.first.c_str());
            }
          }
        }
        else if (m_->body_dofnum[body_id] == 6 &&
                 m_->body_jntadr[body_id] != -1 &&
                 m_->jnt_type[m_->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
        {
          mjtNum *xpos_desired = d_->xpos + 3 * body_id;
          mjtNum *xquat_desired = d_->xquat + 4 * body_id;

          for (const std::string &attribute_name : send_object.second)
          {
            const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              const Json::Value x_json = attribute_data[0];
              const Json::Value y_json = attribute_data[1];
              const Json::Value z_json = attribute_data[2];
              if (!x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                xpos_desired[0] = x_json.asDouble();
                xpos_desired[1] = y_json.asDouble();
                xpos_desired[2] = z_json.asDouble();
              }
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              const Json::Value w_json = attribute_data[0];
              const Json::Value x_json = attribute_data[1];
              const Json::Value y_json = attribute_data[2];
              const Json::Value z_json = attribute_data[3];
              if (!w_json.isNull() && !x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                xquat_desired[0] = w_json.asDouble();
                xquat_desired[1] = x_json.asDouble();
                xquat_desired[2] = y_json.asDouble();
                xquat_desired[3] = z_json.asDouble();
              }
            }
            else if (strcmp(attribute_name.c_str(), "relative_velocity") == 0)
            {
              const Json::Value qvel_lin_x = attribute_data[0];
              const Json::Value qvel_lin_y = attribute_data[1];
              const Json::Value qvel_lin_z = attribute_data[2];
              const Json::Value qvel_ang_x = attribute_data[3];
              const Json::Value qvel_ang_y = attribute_data[4];
              const Json::Value qvel_ang_z = attribute_data[5];
              if (!qvel_lin_x.isNull() && !qvel_lin_y.isNull() && !qvel_lin_z.isNull() && !qvel_ang_x.isNull() && !qvel_ang_y.isNull() && !qvel_ang_z.isNull())
              {
                const int qvel_adr = m_->body_dofadr[body_id];
                d_->qvel[qvel_adr] = qvel_lin_x.asDouble();
                d_->qvel[qvel_adr + 1] = qvel_lin_y.asDouble();
                d_->qvel[qvel_adr + 2] = qvel_lin_z.asDouble();
                d_->qvel[qvel_adr + 3] = qvel_ang_x.asDouble();
                d_->qvel[qvel_adr + 4] = qvel_ang_y.asDouble();
                d_->qvel[qvel_adr + 5] = qvel_ang_z.asDouble();
              }
            }
            else
            {
              mju_warning("Attribute %s not supported for body %s\n", attribute_name.c_str(), send_object.first.c_str());
            }
          }

          const int qpos_adr = m_->jnt_qposadr[m_->body_jntadr[body_id]];
          d_->qpos[qpos_adr] = xpos_desired[0];
          d_->qpos[qpos_adr + 1] = xpos_desired[1];
          d_->qpos[qpos_adr + 2] = xpos_desired[2];
          d_->qpos[qpos_adr + 3] = xquat_desired[0];
          d_->qpos[qpos_adr + 4] = xquat_desired[1];
          d_->qpos[qpos_adr + 5] = xquat_desired[2];
          d_->qpos[qpos_adr + 6] = xquat_desired[3];
        }
        else if (m_->body_dofnum[body_id] == 3 && m_->body_jntadr[body_id] != -1 && m_->jnt_type[m_->body_jntadr[body_id]] == mjtJoint::mjJNT_BALL)
        {
          for (const std::string &attribute_name : send_object.second)
          {
            const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
            if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              const Json::Value w_json = attribute_data[0];
              const Json::Value x_json = attribute_data[1];
              const Json::Value y_json = attribute_data[2];
              const Json::Value z_json = attribute_data[3];

              if (!w_json.isNull() && !x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                const mjtNum xquat_desired[4] = {w_json.asDouble(), x_json.asDouble(), y_json.asDouble(), z_json.asDouble()};
                mjtNum *xquat_current_neg = d_->xquat + 4 * body_id;
                mju_negQuat(xquat_current_neg, xquat_current_neg);

                const int qpos_id = m_->jnt_qposadr[m_->body_jntadr[body_id]];
                mju_mulQuat(d_->qpos + qpos_id, xquat_current_neg, xquat_desired);
              }
            }
            else
            {
              mju_warning("Attribute %s not supported for body %s\n", attribute_name.c_str(), send_object.first.c_str());
            }
          }
        }
      }
      if (joint_id != -1)
      {
        for (const std::string &attribute_name : send_object.second)
        {
          const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
          if ((strcmp(attribute_name.c_str(), "joint_rvalue") == 0 && m_->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
              (strcmp(attribute_name.c_str(), "joint_tvalue") == 0 && m_->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
          {
            const Json::Value v_json = attribute_data[0];
            if (!v_json.isNull())
            {
              const int qpos_id = m_->jnt_qposadr[joint_id];
              d_->qpos[qpos_id] = v_json.asDouble();
            }
          }
          else if ((strcmp(attribute_name.c_str(), "joint_angular_velocity") == 0 && m_->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
                   (strcmp(attribute_name.c_str(), "joint_linear_velocity") == 0 && m_->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
          {
            const Json::Value v_json = attribute_data[0];
            if (!v_json.isNull())
            {
              const int dof_id = m_->jnt_dofadr[joint_id];
              d_->qvel[dof_id] = v_json.asDouble();
            }
          }
          else if ((strcmp(attribute_name.c_str(), "joint_quaternion") == 0 && m_->jnt_type[joint_id] == mjtJoint::mjJNT_BALL))
          {
            const Json::Value w_json = attribute_data[0];
            const Json::Value x_json = attribute_data[1];
            const Json::Value y_json = attribute_data[2];
            const Json::Value z_json = attribute_data[3];

            if (!w_json.isNull() && !x_json.isNull() && !y_json.isNull() && !z_json.isNull())
            {
              const int qpos_adr = m_->jnt_qposadr[joint_id];
              d_->qpos[qpos_adr] = w_json.asDouble();
              d_->qpos[qpos_adr + 1] = x_json.asDouble();
              d_->qpos[qpos_adr + 2] = y_json.asDouble();
              d_->qpos[qpos_adr + 3] = z_json.asDouble();
            }
          }
          else
          {
            mju_warning("Attribute %s not supported for joint %s\n", attribute_name.c_str(), send_object.first.c_str());
          }
        }
      }
      if (actuator_id != -1)
      {
        for (const std::string &attribute_name : send_object.second)
        {
          if (strcmp(attribute_name.c_str(), "cmd_joint_rvalue") == 0 ||
              strcmp(attribute_name.c_str(), "cmd_joint_tvalue") == 0 ||
              strcmp(attribute_name.c_str(), "cmd_joint_angular_velocity") == 0 ||
              strcmp(attribute_name.c_str(), "cmd_joint_linear_velocity") == 0 ||
              strcmp(attribute_name.c_str(), "cmd_joint_torque") == 0 ||
              strcmp(attribute_name.c_str(), "cmd_joint_force") == 0)
          {
            d_->ctrl[actuator_id] = response_meta_data_json["send"][send_object.first][attribute_name][0].asDouble();
          }
          else
          {
            mju_warning("Attribute %s not supported for actuator %s\n", attribute_name.c_str(), send_object.first.c_str());
          }
        }
      }
    }
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
    for (const std::pair<const std::string, std::set<std::string>> &send_object : config_.send_objects)
    {
      const int body_id = mj_name2id(m_, mjtObj::mjOBJ_BODY, send_object.first.c_str());
      const int mocap_id = m_->body_mocapid[body_id];
      // const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, send_object.first.c_str());
      if (body_id != -1)
      {
        const std::string body_name = send_object.first;
        int sensor_id = get_sensor_id(m_, mjOBJ_BODY, body_id);
        if (sensor_id == -1)
        {
          mju_warning("Sensor id for body %s not found\n", body_name.c_str());
          continue;
        }
        if (mocap_id != -1)
        {
          for (const std::string &attribute_name : send_object.second)
          {
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_pos[3 * mocap_id]);
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_pos[3 * mocap_id + 1]);
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_pos[3 * mocap_id + 2]);
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_quat[4 * mocap_id]);
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_quat[4 * mocap_id + 1]);
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_quat[4 * mocap_id + 2]);
              send_data_pairs.emplace_back(sensor_id++, &d_->mocap_quat[4 * mocap_id + 3]);
            }
            else
            {
              mju_warning("Send %s for %s not supported\n", attribute_name.c_str(), body_name.c_str());
            }
          }
        }
        else
        {
          // const int dof_id = m_->body_dofadr[body_id];
          for (const std::string &attribute_name : send_object.second)
          {
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              send_data_pairs.emplace_back(sensor_id++, &d_->xpos[3 * body_id]);
              send_data_pairs.emplace_back(sensor_id++, &d_->xpos[3 * body_id + 1]);
              send_data_pairs.emplace_back(sensor_id++, &d_->xpos[3 * body_id + 2]);
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              send_data_pairs.emplace_back(sensor_id++, &d_->xquat[4 * body_id]);
              send_data_pairs.emplace_back(sensor_id++, &d_->xquat[4 * body_id + 1]);
              send_data_pairs.emplace_back(sensor_id++, &d_->xquat[4 * body_id + 2]);
              send_data_pairs.emplace_back(sensor_id++, &d_->xquat[4 * body_id + 3]);
            }
            // else if (strcmp(attribute_name.c_str(), "force") == 0 &&
            //          m->body_dofnum[body_id] == 6 &&
            //          m->body_jntadr[body_id] != -1 &&
            //          m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
            // {
            //   if (contact_efforts.count(body_id) == 0)
            //   {
            //     contact_efforts[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
            //   }

            //   send_data_vec.emplace_back(&contact_efforts[body_id][0]);
            //   send_data_vec.emplace_back(&contact_efforts[body_id][1]);
            //   send_data_vec.emplace_back(&contact_efforts[body_id][2]);
            // }
            // else if (strcmp(attribute_name.c_str(), "torque") == 0 &&
            //          m->body_dofnum[body_id] == 6 &&
            //          m->body_jntadr[body_id] != -1 &&
            //          m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
            // {
            //   if (contact_efforts.count(body_id) == 0)
            //   {
            //     contact_efforts[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
            //   }

            //   send_data_vec.emplace_back(&contact_efforts[body_id][3]);
            //   send_data_vec.emplace_back(&contact_efforts[body_id][4]);
            //   send_data_vec.emplace_back(&contact_efforts[body_id][5]);
            // }
            // else if (strcmp(attribute_name.c_str(), "relative_velocity") == 0 &&
            //          m->body_dofnum[body_id] == 6 &&
            //          m->body_jntadr[body_id] != -1 &&
            //          m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
            // {
            //   send_data_vec.emplace_back(&d->qvel[dof_id]);
            //   send_data_vec.emplace_back(&d->qvel[dof_id + 1]);
            //   send_data_vec.emplace_back(&d->qvel[dof_id + 2]);
            //   send_data_vec.emplace_back(&d->qvel[dof_id + 3]);
            //   send_data_vec.emplace_back(&d->qvel[dof_id + 4]);
            //   send_data_vec.emplace_back(&d->qvel[dof_id + 5]);
            // }
            // else if (strcmp(attribute_name.c_str(), "odometric_velocity") == 0 &&
            //          m->body_dofnum[body_id] <= 6 &&
            //          m->body_jntadr[body_id] != -1)
            // {
            //   odom_velocities[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
            //   send_data_vec.emplace_back(&odom_velocities[body_id][0]);
            //   send_data_vec.emplace_back(&odom_velocities[body_id][1]);
            //   send_data_vec.emplace_back(&odom_velocities[body_id][2]);
            //   send_data_vec.emplace_back(&odom_velocities[body_id][3]);
            //   send_data_vec.emplace_back(&odom_velocities[body_id][4]);
            //   send_data_vec.emplace_back(&odom_velocities[body_id][5]);
            // }
            else
            {
              printf("Send %s for %s not supported\n", attribute_name.c_str(), body_name.c_str());
            }
          }
        }
      }
      // else if (joint_id != -1)
      // {
      //   const std::string joint_name = send_object.first;
      //   const int qpos_id = m->jnt_qposadr[joint_id];
      //   const int dof_id = m->jnt_dofadr[joint_id];
      //   for (const std::string &attribute_name : send_object.second)
      //   {
      //     if ((strcmp(attribute_name.c_str(), "joint_rvalue") == 0 &&
      //          m->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
      //         (strcmp(attribute_name.c_str(), "joint_tvalue") == 0 &&
      //          m->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
      //     {
      //       send_data_vec.emplace_back(&d->qpos[qpos_id]);
      //     }
      //     else if ((strcmp(attribute_name.c_str(), "joint_angular_velocity") == 0 &&
      //               m->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
      //              (strcmp(attribute_name.c_str(), "joint_linear_velocity") == 0 &&
      //               m->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
      //     {
      //       send_data_vec.emplace_back(&d->qvel[dof_id]);
      //     }
      //     else if ((strcmp(attribute_name.c_str(), "joint_torque") == 0 &&
      //               m->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
      //              (strcmp(attribute_name.c_str(), "joint_force") == 0 &&
      //               m->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
      //     {
      //       send_data_vec.emplace_back(&d->qfrc_inverse[dof_id]);
      //     }
      //     else if (strcmp(attribute_name.c_str(), "joint_position") == 0)
      //     {
      //       printf("Send %s for %s not supported yet\n", attribute_name.c_str(), joint_name.c_str());
      //     }
      //     else if (strcmp(attribute_name.c_str(), "joint_quaternion") == 0 &&
      //              m->jnt_type[joint_id] == mjtJoint::mjJNT_BALL)
      //     {
      //       send_data_vec.emplace_back(&d->qpos[qpos_id]);
      //       send_data_vec.emplace_back(&d->qpos[qpos_id + 1]);
      //       send_data_vec.emplace_back(&d->qpos[qpos_id + 2]);
      //       send_data_vec.emplace_back(&d->qpos[qpos_id + 3]);
      //     }
      //     else
      //     {
      //       printf("Send %s for %s not supported\n", attribute_name.c_str(), joint_name.c_str());
      //     }
      //   }
      // }
    }

    // for (const std::pair<std::string, std::set<std::string>> &receive_object : receive_objects)
    // {
    //   const int body_id = mj_name2id(m, mjtObj::mjOBJ_BODY, receive_object.first.c_str());
    //   const int mocap_id = m->body_mocapid[body_id];
    //   const int joint_id = mj_name2id(m, mjtObj::mjOBJ_JOINT, receive_object.first.c_str());
    //   const int actuator_id = mj_name2id(m, mjtObj::mjOBJ_ACTUATOR, receive_object.first.c_str());
    //   if (body_id != -1)
    //   {
    //     const std::string body_name = receive_object.first;
    //     if (mocap_id != -1)
    //     {
    //       for (const std::string &attribute_name : receive_object.second)
    //       {
    //         if (strcmp(attribute_name.c_str(), "position") == 0)
    //         {
    //           receive_data_vec.emplace_back(&d->mocap_pos[3 * mocap_id]);
    //           receive_data_vec.emplace_back(&d->mocap_pos[3 * mocap_id + 1]);
    //           receive_data_vec.emplace_back(&d->mocap_pos[3 * mocap_id + 2]);
    //         }
    //         else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
    //         {
    //           receive_data_vec.emplace_back(&d->mocap_quat[4 * mocap_id]);
    //           receive_data_vec.emplace_back(&d->mocap_quat[4 * mocap_id + 1]);
    //           receive_data_vec.emplace_back(&d->mocap_quat[4 * mocap_id + 2]);
    //           receive_data_vec.emplace_back(&d->mocap_quat[4 * mocap_id + 3]);
    //         }
    //         else
    //         {
    //           printf("Send %s for %s not supported\n", attribute_name.c_str(), body_name.c_str());
    //         }
    //       }
    //     }
    //     else
    //     {
    //       const int dof_id = m->body_dofadr[body_id];
    //       for (const std::string &attribute_name : receive_object.second)
    //       {
    //         if (strcmp(attribute_name.c_str(), "position") == 0 &&
    //             m->body_dofnum[body_id] == 6 &&
    //             m->body_jntadr[body_id] != -1 &&
    //             m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
    //         {
    //           int qpos_id = m->jnt_qposadr[m->body_jntadr[body_id]];
    //           receive_data_vec.emplace_back(&d->qpos[qpos_id]);
    //           receive_data_vec.emplace_back(&d->qpos[qpos_id + 1]);
    //           receive_data_vec.emplace_back(&d->qpos[qpos_id + 2]);
    //         }
    //         else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
    //         {
    //           if (m->body_dofnum[body_id] == 6 &&
    //               m->body_jntadr[body_id] != -1 &&
    //               m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE)
    //           {
    //             int qpos_id = m->jnt_qposadr[m->body_jntadr[body_id]];
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 3]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 4]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 5]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 6]);
    //           }
    //           else if (m->body_dofnum[body_id] == 3 &&
    //                    m->body_jntadr[body_id] != -1 &&
    //                    m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_BALL)
    //           {
    //             int qpos_id = m->jnt_qposadr[m->body_jntadr[body_id]];
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 1]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 2]);
    //             receive_data_vec.emplace_back(&d->qpos[qpos_id + 3]);
    //           }
    //         }
    //         else if (strcmp(attribute_name.c_str(), "force") == 0)
    //         {
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id]);
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id + 1]);
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id + 2]);
    //         }
    //         else if (strcmp(attribute_name.c_str(), "torque") == 0)
    //         {
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id + 3]);
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id + 4]);
    //           receive_data_vec.emplace_back(&d->xfrc_applied[6 * body_id + 5]);
    //         }
    //         else if (strcmp(attribute_name.c_str(), "relative_velocity") == 0 &&
    //                  m->body_dofnum[body_id] == 6 &&
    //                  m->body_jntadr[body_id] != -1 &&
    //                  m->jnt_type[m->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE &&
    //                  odom_velocities.count(body_id) == 0)
    //         {
    //           receive_data_vec.emplace_back(&d->qvel[dof_id]);
    //           receive_data_vec.emplace_back(&d->qvel[dof_id + 1]);
    //           receive_data_vec.emplace_back(&d->qvel[dof_id + 2]);
    //           receive_data_vec.emplace_back(&d->qvel[dof_id + 3]);
    //           receive_data_vec.emplace_back(&d->qvel[dof_id + 4]);
    //           receive_data_vec.emplace_back(&d->qvel[dof_id + 5]);
    //         }
    //         else if (strcmp(attribute_name.c_str(), "odometric_velocity") == 0)
    //         {
    //           if (m->body_dofnum[body_id] <= 6 &&
    //               m->body_jntadr[body_id] != -1 &&
    //               odom_velocities.count(body_id) == 0)
    //           {
    //             odom_velocities[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][0]);
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][1]);
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][2]);
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][3]);
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][4]);
    //             receive_data_vec.emplace_back(&odom_velocities[body_id][5]);
    //           }
    //         }
    //       }
    //     }
    //   }
    //   else if (joint_id != -1)
    //   {
    //     const std::string joint_name = receive_object.first;
    //     const int qpos_id = m->jnt_qposadr[joint_id];
    //     const int dof_id = m->jnt_dofadr[joint_id];
    //     for (const std::string &attribute_name : receive_object.second)
    //     {
    //       if ((strcmp(attribute_name.c_str(), "cmd_joint_torque") == 0 && m->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE) ||
    //           (strcmp(attribute_name.c_str(), "cmd_joint_force") == 0 && m->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE))
    //       {
    //         receive_data_vec.emplace_back(&d->qfrc_applied[dof_id]);
    //       }
    //       else
    //       {
    //         printf("Receive %s for %s not supported\n", attribute_name.c_str(), joint_name.c_str());
    //       }
    //     }
    //   }
    //   else if (actuator_id != -1)
    //   {
    //     const std::string actuator_name = receive_object.first;
    //     for (const std::string &attribute_name : receive_object.second)
    //     {
    //       if (strcmp(attribute_name.c_str(), "cmd_joint_rvalue") == 0 ||
    //           strcmp(attribute_name.c_str(), "cmd_joint_tvalue") == 0 ||
    //           strcmp(attribute_name.c_str(), "cmd_joint_angular_velocity") == 0 ||
    //           strcmp(attribute_name.c_str(), "cmd_joint_linear_velocity") == 0 ||
    //           strcmp(attribute_name.c_str(), "cmd_joint_torque") == 0 ||
    //           strcmp(attribute_name.c_str(), "cmd_joint_force") == 0)
    //       {
    //         receive_data_vec.emplace_back(&d->ctrl[actuator_id]);
    //       }
    //       else
    //       {
    //         printf("Receive %s for %s not supported\n", attribute_name.c_str(), actuator_name.c_str());
    //       }
    //     }
    //   }
    // }
  }

  void MultiverseConnector::bind_send_data()
  {
    for (const std::pair<int, double*>& send_data_pair : send_data_pairs)
    {
      const int sensor_id = send_data_pair.first;
      const double *data = send_data_pair.second;
      d_->sensordata[sensor_id] = *data;
    }
  }

  void MultiverseConnector::bind_receive_data()
  {
    
  }

} // namespace mujoco::plugin::multiverse_connector
