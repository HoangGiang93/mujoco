// Tests for multiverse_connector plugins.

#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir()
{
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string
  {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success)
    {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size)
      {
        success = true;
      }
      else if (written == buf_size)
      {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
      else
      {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf)
    {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size))
    {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char *path = buf.get();
#else
  const char *path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string
  {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success)
    {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size)
      {
        realpath.get()[written] = '\0';
        success = true;
      }
      else if (written == -1)
      {
        if (errno == EINVAL)
        {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      }
      else
      {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty())
  {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i)
  {
    if (realpath.c_str()[i] == kPathSep)
    {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
const std::string sep = "\\";
#else
const std::string sep = "/";
#endif

void load_plugins()
{
  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char *filename, int first, int count)
                          {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
}

namespace mujoco
{
  namespace
  {

    using MulitverseConnectorPluginTest = MujocoTest;

    TEST_F(MulitverseConnectorPluginTest, ValidAttributes)
    {
      load_plugins();
      const std::string xml_path = GetTestDataFilePath("test/plugin/multiverse_connector/testdata/multiverse_connector_test.xml");
      char error[1024] = {0};
      mjModel *model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
      ASSERT_THAT(model, testing::NotNull()) << error;
      EXPECT_THAT(model->sensor_dim[0], 10);
      mj_deleteModel(model);
    }

  } // namespace
} // namespace mujoco
