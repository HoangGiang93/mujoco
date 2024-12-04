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

bool is_attribute_valid(const std::string &attr, const int obj_type, int &attr_size)
{
  attr_size = 0;
  switch (obj_type)
  {
  case mjOBJ_BODY:
  {
    if (strcmp(attr.c_str(), "position") == 0)
    {
      attr_size = 3;
      return true;
    }
    else if (strcmp(attr.c_str(), "quaternion") == 0)
    {
      attr_size = 4;
      return true;
    }
    else if (strcmp(attr.c_str(), "relative_velocity") == 0)
    {
      attr_size = 6;
      return true;
    }
    else if (strcmp(attr.c_str(), "force") == 0)
    {
      attr_size = 3;
      return true;
    }
    else if (strcmp(attr.c_str(), "torque") == 0)
    {
      attr_size = 3;
      return true;
    }
    return false;
  }
  case mjOBJ_JOINT:
  {
    const std::set<const char *> joint_attributes = {"joint_rvalue", "joint_tvalue", "joint_angular_velocity", "joint_linear_velocity", "joint_torque", "joint_force"};
    if (std::find(joint_attributes.begin(), joint_attributes.end(), attr) != joint_attributes.end())
    {
      attr_size = 1;
      return true;
    }
    return false;
  }
  case mjOBJ_ACTUATOR:
  {
    const std::set<const char *> actuator_attributes = {"cmd_joint_rvalue", "cmd_joint_tvalue", "cmd_joint_angular_velocity", "cmd_joint_linear_velocity", "cmd_joint_torque", "cmd_joint_force"};
    if (std::find(actuator_attributes.begin(), actuator_attributes.end(), attr) != actuator_attributes.end())
    {
      attr_size = 1;
      return true;
    }
    return false;
  }
  default:
    mju_warning("Object type %d is not supported\n", obj_type);
    return false;
  }
}

namespace mujoco::plugin::multiverse_connector
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

  MultiverseConnector *MultiverseConnector::Create(const mjModel *m, mjData *d, int instance)
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
      return new MultiverseConnector(config, m, d);
    }

    std::string send_str = GetStringAttr(m, instance, send);
    boost::replace_all(send_str, "'", "\"");
    Json::Value send_json = string_to_json(send_str);
    Json::Reader reader;
    if (!send_json.empty())
    {
      for (const int sensor_id : sensor_ids)
      {
        const int obj_type = m->sensor_objtype[sensor_id];
        const int obj_id = m->sensor_objid[sensor_id];
        const char *obj_name = mj_id2name(m, obj_type, obj_id);
        if (!obj_name)
        {
          mju_warning("Object id %d of type %d must have a name\n", obj_id, obj_type);
          continue;
        }
        config.send_objects[obj_name] = {};
        for (const Json::Value &attribute_json : send_json[obj_name])
        {
          const std::string attribute_name = attribute_json.asString();
          int attr_size = 0;
          if (is_attribute_valid(attribute_name, obj_type, attr_size))
          {
            config.send_objects[obj_name].insert(attribute_name);
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

    return new MultiverseConnector(config, m, d);
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

      const int obj_type = m->sensor_objtype[sensor_id];
      if (obj_type == mjOBJ_UNKNOWN)
      {
        return 10; // Special case for unknown object type
      }
      const int obj_id = m->sensor_objid[sensor_id];
      const char *obj_name = mj_id2name(m, obj_type, obj_id);
      if (!obj_name)
      {
        mju_warning("Object id %d of type %d must have a name\n", obj_id, obj_type);
        return 0;
      }
      if (!send_json.isMember(obj_name))
      {
        mju_warning_s("Object %s is not in the send list\n", obj_name);
        return 0;
      }
      int nsensordata = 0;
      for (const Json::Value &attribute_json : send_json[obj_name])
      {
        const std::string attribute_name = attribute_json.asString();
        int attr_size = 0;
        if (is_attribute_valid(attribute_name, obj_type, attr_size))
        {
          nsensordata += attr_size;
        }
      }
      return nsensordata;
    };

    plugin.init = +[](const mjModel *m, mjData *d, int instance)
    {
      MultiverseConnector *multiverse_connector = MultiverseConnector::Create(m, d, instance);
      if (multiverse_connector == nullptr)
      {
        return -1;
      }
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(multiverse_connector);
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
      if (body_id != -1 || joint_id != -1 || actuator_id != -1)
      {
        const std::string object_name = send_object.first;
        for (const std::string &attribute_name : send_object.second)
        {
          request_meta_data_json["send"][object_name].append(attribute_name);
        }
      }
    }

    for (const std::pair<const std::string, std::set<std::string>> &receive_object : config_.receive_objects)
    {
      const int body_id = mj_name2id(m_, mjtObj::mjOBJ_BODY, receive_object.first.c_str());
      const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, receive_object.first.c_str());
      const int actuator_id = mj_name2id(m_, mjtObj::mjOBJ_ACTUATOR, receive_object.first.c_str());
      if (body_id != -1 || joint_id != -1 || actuator_id != -1)
      {
        const std::string object_name = receive_object.first;
        for (const std::string &attribute_name : receive_object.second)
        {
          request_meta_data_json["receive"][object_name].append(attribute_name);
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
        const bool is_static = mocap_id != -1;
        const bool is_free = m_->body_dofnum[body_id] == 6 && m_->body_jntadr[body_id] != -1 && m_->jnt_type[m_->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE;
        const bool is_hanging = m_->body_dofnum[body_id] == 3 && m_->body_jntadr[body_id] != -1 && m_->jnt_type[m_->body_jntadr[body_id]] == mjtJoint::mjJNT_BALL;
        if (is_static)
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
          }
        }
        else if (is_free)
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
        else if (is_hanging)
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
          }
        }
        if (!is_static)
        {
          for (const std::string &attribute_name : send_object.second)
          {
            const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
            if (strcmp(attribute_name.c_str(), "force") == 0)
            {
              const Json::Value x_json = attribute_data[0];
              const Json::Value y_json = attribute_data[1];
              const Json::Value z_json = attribute_data[2];
              if (!x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                d_->xfrc_applied[6 * body_id] = x_json.asDouble();
                d_->xfrc_applied[6 * body_id + 1] = y_json.asDouble();
                d_->xfrc_applied[6 * body_id + 2] = z_json.asDouble();
              }
            }
            else if (strcmp(attribute_name.c_str(), "torque") == 0)
            {
              const Json::Value x_json = attribute_data[0];
              const Json::Value y_json = attribute_data[1];
              const Json::Value z_json = attribute_data[2];
              if (!x_json.isNull() && !y_json.isNull() && !z_json.isNull())
              {
                d_->xfrc_applied[6 * body_id + 3] = x_json.asDouble();
                d_->xfrc_applied[6 * body_id + 4] = y_json.asDouble();
                d_->xfrc_applied[6 * body_id + 5] = z_json.asDouble();
              }
            }
          }
        }
      }
      if (joint_id != -1)
      {
        const bool is_revolute_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE;
        const bool is_prismatic_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE;
        const bool is_ball_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_BALL;
        for (const std::string &attribute_name : send_object.second)
        {
          const Json::Value attribute_data = response_meta_data_json["send"][send_object.first][attribute_name];
          if ((strcmp(attribute_name.c_str(), "joint_rvalue") == 0 && is_revolute_joint) ||
              (strcmp(attribute_name.c_str(), "joint_tvalue") == 0 && is_prismatic_joint))
          {
            const Json::Value v_json = attribute_data[0];
            if (!v_json.isNull())
            {
              const int qpos_id = m_->jnt_qposadr[joint_id];
              d_->qpos[qpos_id] = v_json.asDouble();
            }
          }
          else if ((strcmp(attribute_name.c_str(), "joint_angular_velocity") == 0 && is_revolute_joint) ||
                   (strcmp(attribute_name.c_str(), "joint_linear_velocity") == 0 && is_prismatic_joint))
          {
            const Json::Value v_json = attribute_data[0];
            if (!v_json.isNull())
            {
              const int dof_id = m_->jnt_dofadr[joint_id];
              d_->qvel[dof_id] = v_json.asDouble();
            }
          }
          else if ((strcmp(attribute_name.c_str(), "joint_torque") == 0 && is_revolute_joint) ||
                   (strcmp(attribute_name.c_str(), "joint_force") == 0 && is_prismatic_joint))
          {
            const Json::Value v_json = attribute_data[0];
            if (!v_json.isNull())
            {
              const int dof_id = m_->jnt_dofadr[joint_id];
              d_->qfrc_applied[dof_id] = v_json.asDouble();
            }
          }
          else if ((strcmp(attribute_name.c_str(), "joint_quaternion") == 0 && is_ball_joint))
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
    send_data_pairs.clear();

    for (std::pair<const int, mjtNum *> &contact_effort : contact_efforts)
    {
      free(contact_effort.second);
    }
    contact_efforts.clear();
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
      const int joint_id = mj_name2id(m_, mjtObj::mjOBJ_JOINT, send_object.first.c_str());
      if (body_id != -1)
      {
        const std::string body_name = send_object.first;
        int sensor_id = get_sensor_id(m_, mjOBJ_BODY, body_id);
        if (sensor_id == -1)
        {
          mju_warning("Sensor id for body %s not found\n", body_name.c_str());
          continue;
        }
        int sensor_adr = m_->sensor_adr[sensor_id];
        if (mocap_id != -1)
        {
          for (const std::string &attribute_name : send_object.second)
          {
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_pos[3 * mocap_id]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_pos[3 * mocap_id + 1]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_pos[3 * mocap_id + 2]);
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_quat[4 * mocap_id]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_quat[4 * mocap_id + 1]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_quat[4 * mocap_id + 2]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->mocap_quat[4 * mocap_id + 3]);
            }
            else
            {
              mju_warning("Send %s for %s not supported\n", attribute_name.c_str(), body_name.c_str());
            }
          }
        }
        else
        {
          const int dof_id = m_->body_dofadr[body_id];
          const bool is_free = m_->body_dofnum[body_id] == 6 && m_->body_jntadr[body_id] != -1 && m_->jnt_type[m_->body_jntadr[body_id]] == mjtJoint::mjJNT_FREE;
          for (const std::string &attribute_name : send_object.second)
          {
            if (strcmp(attribute_name.c_str(), "position") == 0)
            {
              send_data_pairs.emplace_back(sensor_adr++, &d_->xpos[3 * body_id]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->xpos[3 * body_id + 1]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->xpos[3 * body_id + 2]);
            }
            else if (strcmp(attribute_name.c_str(), "quaternion") == 0)
            {
              send_data_pairs.emplace_back(sensor_adr++, &d_->xquat[4 * body_id]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->xquat[4 * body_id + 1]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->xquat[4 * body_id + 2]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->xquat[4 * body_id + 3]);
            }
            else if (strcmp(attribute_name.c_str(), "force") == 0 && is_free)
            {
              if (contact_efforts.count(body_id) == 0)
              {
                contact_efforts[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
              }

              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][0]);
              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][1]);
              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][2]);
            }
            else if (strcmp(attribute_name.c_str(), "torque") == 0 && is_free)
            {
              if (contact_efforts.count(body_id) == 0)
              {
                contact_efforts[body_id] = (mjtNum *)calloc(6, sizeof(mjtNum));
              }

              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][3]);
              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][4]);
              send_data_pairs.emplace_back(sensor_adr++, &contact_efforts[body_id][5]);
            }
            else if (strcmp(attribute_name.c_str(), "relative_velocity") == 0 && is_free)
            {
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id + 1]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id + 2]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id + 3]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id + 4]);
              send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id + 5]);
            }
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
          }
        }
      }
      else if (joint_id != -1)
      {
        const std::string joint_name = send_object.first;
        const int qpos_id = m_->jnt_qposadr[joint_id];
        const int dof_id = m_->jnt_dofadr[joint_id];
        int sensor_id = get_sensor_id(m_, mjOBJ_JOINT, joint_id);
        if (sensor_id == -1)
        {
          mju_warning("Sensor id for joint %s not found\n", joint_name.c_str());
          continue;
        }
        int sensor_adr = m_->sensor_adr[sensor_id];
        for (const std::string &attribute_name : send_object.second)
        {
          const bool is_revolute_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_HINGE;
          const bool is_prismatic_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_SLIDE;
          const bool is_ball_joint = m_->jnt_type[joint_id] == mjtJoint::mjJNT_BALL;
          if ((strcmp(attribute_name.c_str(), "joint_rvalue") == 0 && is_revolute_joint) ||
              (strcmp(attribute_name.c_str(), "joint_tvalue") == 0 && is_prismatic_joint))
          {
            send_data_pairs.emplace_back(sensor_adr++, &d_->qpos[qpos_id]);
          }
          else if ((strcmp(attribute_name.c_str(), "joint_angular_velocity") == 0 && is_revolute_joint) ||
                   (strcmp(attribute_name.c_str(), "joint_linear_velocity") == 0 && is_prismatic_joint))
          {
            send_data_pairs.emplace_back(sensor_adr++, &d_->qvel[dof_id]);
          }
          else if ((strcmp(attribute_name.c_str(), "joint_torque") == 0 && is_revolute_joint) ||
                   (strcmp(attribute_name.c_str(), "joint_force") == 0 && is_prismatic_joint))
          {
            send_data_pairs.emplace_back(sensor_adr++, &d_->qfrc_inverse[dof_id]);
          }
          else if (strcmp(attribute_name.c_str(), "joint_position") == 0)
          {
            mju_warning("Send %s for %s not supported yet\n", attribute_name.c_str(), joint_name.c_str());
          }
          else if (strcmp(attribute_name.c_str(), "joint_quaternion") == 0 && is_prismatic_joint && is_ball_joint)
          {
            send_data_pairs.emplace_back(sensor_adr++, &d_->qpos[qpos_id]);
            send_data_pairs.emplace_back(sensor_adr++, &d_->qpos[qpos_id + 1]);
            send_data_pairs.emplace_back(sensor_adr++, &d_->qpos[qpos_id + 2]);
            send_data_pairs.emplace_back(sensor_adr++, &d_->qpos[qpos_id + 3]);
          }
          else
          {
            mju_warning("Send %s for %s not supported\n", attribute_name.c_str(), joint_name.c_str());
          }
        }
      }
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
    *world_time = d_->time;
    if (send_data_pairs.size() != send_buffer.buffer_double.size)
    {
      mju_warning("Mismatch between send_data_pairs [%zd] and send_buffer.buffer_double.size [%zd]\n", send_data_pairs.size(), send_buffer.buffer_double.size);
      return;
    }

    mj_markStack(d_);
    for (std::pair<const int, mjtNum *> &contact_effort : contact_efforts)
    {
      mjtNum *jac = mj_stackAllocNum(d_, 6 * m_->nv);
      mj_jacBodyCom(m_, d_, jac, jac + 3 * m_->nv, contact_effort.first);
      mju_mulMatVec(contact_effort.second, jac, d_->qfrc_constraint, 6, m_->nv);
    }
    mj_freeStack(d_);

    for (size_t i = 0; i < send_buffer.buffer_double.size; i++)
    {
      const int sensor_id = send_data_pairs[i].first;
      const double *data = send_data_pairs[i].second;
      d_->sensordata[sensor_id] = *data;
      send_buffer.buffer_double.data[i] = *data;
    }
  }

  void MultiverseConnector::bind_receive_data()
  {
  }

} // namespace mujoco::plugin::multiverse_connector
