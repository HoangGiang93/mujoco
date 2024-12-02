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

#ifndef MUJOCO_PLUGIN_MULTIVERSE_CONNECTOR_H_
#define MUJOCO_PLUGIN_MULTIVERSE_CONNECTOR_H_

#include <memory>
#include <vector>
#include <map>
#include <string>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::multiverse_connector {

struct MultiverseConfig {
  std::string server = "tcp://127.0.0.1:7000";
  std::string client = "tcp://127.0.0.1:7500";
  std::map<std::string, std::vector<std::string>> send_objects = {};
  std::map<std::string, std::vector<std::string>> receive_objects = {};
};

// An multiverse_connector plugin which implements configurable MULTIVERSE_CONNECTOR control.
class MultiverseConnector {
 public:
  // Returns an instance of MultiverseConnector. The result can be null in case of
  // misconfiguration.
  static std::unique_ptr<MultiverseConnector> Create(const mjModel* m, int instance);

  // Returns the number of state variables for the plugin instance
  static int StateSize(const mjModel* m, int instance);

  // Resets the C++ MultiverseConnector instance's state.
  // plugin_state is a C array pointer into mjData->plugin_state, with a size
  // equal to the value returned from StateSize.
  void Reset(mjtNum* plugin_state);

  // Computes the rate of change for activation variables
  // void ActDot(const mjModel* m, mjData* d, int instance) const;

  // Idempotent computation which updates d->actuator_force and the internal
  // state of the class. Called after ActDot.
  void Compute(const mjModel* m, mjData* d, int instance);

  // Updates plugin state.
  void Advance(const mjModel* m, mjData* d, int instance) const;

  // Adds the MULTIVERSE_CONNECTOR plugin to the global registry of MuJoCo plugins.
  static void RegisterPlugin();

 private:
  MultiverseConnector(MultiverseConfig config);

  // Returns the expected number of activation variables for the instance.
  // static int ActDim(const mjModel* m, int instance, int actuator_id);

  // struct State {
  //   mjtNum previous_ctrl = 0;
  //   // if using slew rate limits, mjData.act will contain an activation variable
  //   // with the last ctrl value. If `false`, that value should be ignored,
  //   // because it hasn't been set yet.
  //   bool previous_ctrl_exists = false;
  //   mjtNum integral = 0;
  // };
  // Reads data from d->act and returns it as a State struct.
  // State GetState(const mjModel* m, mjData* d, int actuator_idx) const;

  // Returns the MULTIVERSE_CONNECTOR setpoint, which is normally d->ctrl, but can be d->act for
  // actuators with dyntype != none.
  // mjtNum GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
  //                const State& state, bool actearly) const;

  MultiverseConfig config_;
  // set of multiverse_connector IDs controlled by this plugin instance.
  // std::vector<int> actuators_;
};

}  // namespace mujoco::plugin::multiverse_connector

#endif  // MUJOCO_PLUGIN_MULTIVERSE_CONNECTOR_H_
