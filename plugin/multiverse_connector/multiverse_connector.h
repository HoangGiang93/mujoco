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

#include "multiverse_client_json.h"

#include <set>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::multiverse_connector
{
  struct MultiverseConfig
  {
    std::string server_host = "tcp://127.0.0.1";
    std::string server_port = "7000";
    std::string client_port = "7500";
    std::string world_name = "world";
    std::string simulation_name = "mujoco_simulation";
    std::map<std::string, std::set<std::string>> send_objects = {};
    std::map<std::string, std::set<std::string>> receive_objects = {};
  };

  // An multiverse_connector plugin which implements configurable MULTIVERSE_CONNECTOR control.
  class MultiverseConnector : public MultiverseClientJson
  {
  public:
    // Returns an instance of MultiverseConnector. The result can be null in case of
    // misconfiguration.
    static MultiverseConnector *Create(const mjModel *m, mjData *d, int instance);

    // Returns the number of state variables for the plugin instance
    static int StateSize(const mjModel *m, int instance);

    // Resets the C++ MultiverseConnector instance's state.
    // plugin_state is a C array pointer into mjData->plugin_state, with a size
    // equal to the value returned from StateSize.
    void Reset(mjtNum *plugin_state);

    // Computes the rate of change for activation variables
    // void ActDot(const mjModel* m, mjData* d, int instance) const;

    // Idempotent computation which updates d->actuator_force and the internal
    // state of the class. Called after ActDot.
    void Compute(const mjModel *m, mjData *d, int instance);

    // Updates plugin state.
    void Advance(const mjModel *m, mjData *d, int instance) const;

    // Adds the MULTIVERSE_CONNECTOR plugin to the global registry of MuJoCo plugins.
    static void RegisterPlugin();

  private:
    MultiverseConnector(MultiverseConfig config, const mjModel *m, mjData *d);

    MultiverseConfig config_;

  private:
    mjModel *m_ = nullptr;

    mjData *d_ = nullptr;

    std::vector<mjtNum*> send_data_vec;

    std::vector<mjtNum*> receive_data_vec;

    std::map<int, mjtNum *> contact_efforts;

  private:
    void start_connect_to_server_thread() override;

    void wait_for_connect_to_server_thread_finish() override;

    void start_meta_data_thread() override;

    void wait_for_meta_data_thread_finish() override;

    bool init_objects(bool from_request_meta_data = false) override;

    void bind_request_meta_data() override;

    void bind_api_callbacks() override;

    void bind_api_callbacks_response() override;

    void bind_response_meta_data() override;

    void init_send_and_receive_data() override;

    void bind_send_data() override;

    void bind_receive_data() override;

    void clean_up() override;

    void reset() override;
  };

} // namespace mujoco::plugin::multiverse_connector

#endif // MUJOCO_PLUGIN_MULTIVERSE_CONNECTOR_H_
