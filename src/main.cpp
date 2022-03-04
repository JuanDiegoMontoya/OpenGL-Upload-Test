#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <ranges>
#include <cstdio>
#include <atomic>
#include <array>
#include <utility>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define METHOD 2
constexpr size_t VEC_SIZE = 256;

const char* const gVertexSource = R"(
#version 460 core

layout(location = 0) out vec2 v_uv;

void main()
{
  vec2 pos = vec2(gl_VertexID == 0, gl_VertexID == 2);
  v_uv = pos.xy * 2.0;
  gl_Position = vec4(pos * 4.0 - 1.0, 0.0, 1.0);
}
)";

const char* const gFragmentSource = R"(
#version 460 core

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec2 v_uv;

layout(binding = 0) uniform sampler2D tex;

void main()
{
  o_color = vec4(texture(tex, v_uv).rgb, 1.0);
}
)";

template<typename T, size_t N>
class CheapVector
{
public:
  constexpr T& operator[](size_t i)
  {
    return array_[i];
  }

  constexpr const T& operator[](size_t i) const
  {
    return array_[i];
  }

  template<class... Args>
  constexpr void emplace_back(Args&&... args)
  {
    auto idx = size_.fetch_add(1);
    new(&array_[idx]) T(std::forward<Args>(args)...);
  }

  constexpr size_t size() const
  {
    return size_;
  }

  constexpr bool empty() const
  {
    return size_ == 0;
  }

  auto begin()
  {
    return array_.begin();
  }

  auto end()
  {
    return array_.begin() + size_;
  }

  void set_size(size_t s)
  {
    size_ = s;
  }

private:
  std::array<T, N> array_;
  std::atomic<size_t> size_{};
};

class Timer
{
  using millisecond_t = std::chrono::duration<double, std::ratio<1, 1'000>>;
  using myclock_t = std::chrono::high_resolution_clock;
  using timepoint_t = std::chrono::time_point<myclock_t>;
public:
  Timer()
  {
    timepoint_ = myclock_t::now();
  }

  void Reset()
  {
    timepoint_ = myclock_t::now();
  }

  double Elapsed_ms() const
  {
    timepoint_t beg_ = timepoint_;
    return std::chrono::duration_cast<millisecond_t>(myclock_t::now() - beg_).count();
  }

private:
  timepoint_t timepoint_;
};

GLuint CompileShader(GLenum stage, std::string_view source)
{
  auto sourceStr = std::string(source);
  const GLchar* strings = sourceStr.c_str();

  GLuint shader = glCreateShader(stage);
  glShaderSource(shader, 1, &strings, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    GLsizei infoLength = 512;
    std::string infoLog(infoLength + 1, '\0');
    glGetShaderInfoLog(shader, infoLength, nullptr, infoLog.data());

    throw std::runtime_error(infoLog);
  }

  return shader;
}

void LinkProgram(GLuint program)
{
  glLinkProgram(program);
  GLsizei length = 512;

  GLint success{};
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success)
  {
    std::string infoLog(length + 1, '\0');
    glGetProgramInfoLog(program, length, nullptr, infoLog.data());

    throw std::runtime_error(infoLog);
  }
}

GLuint CompileVertexFragmentProgram(std::string_view vs, std::string_view fs)
{
  GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vs);
  GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fs);

  GLuint program = glCreateProgram();

  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  try { LinkProgram(program); }
  catch (std::runtime_error& e) { glDeleteProgram(program); throw e; }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  return program;
}

std::string LoadFile(std::string_view path, std::ios_base::openmode mode = 0)
{
  std::ifstream file{ path.data(), std::ios_base::in | mode };
  return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

static void GLAPIENTRY glErrorCallback(
  GLenum source,
  GLenum type,
  GLuint id,
  GLenum severity,
  [[maybe_unused]] GLsizei length,
  const GLchar* message,
  [[maybe_unused]] const void* userParam)
{
  // ignore insignificant error/warning codes
  if (id == 131169 || id == 131185 || id == 131218 || id == 131204 || id == 0
    )//|| id == 131188 || id == 131186)
    return;

  std::stringstream errStream;
  errStream << "OpenGL Debug message (" << id << "): " << message << '\n';

  switch (source)
  {
  case GL_DEBUG_SOURCE_API:             errStream << "Source: API"; break;
  case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   errStream << "Source: Window Manager"; break;
  case GL_DEBUG_SOURCE_SHADER_COMPILER: errStream << "Source: Shader Compiler"; break;
  case GL_DEBUG_SOURCE_THIRD_PARTY:     errStream << "Source: Third Party"; break;
  case GL_DEBUG_SOURCE_APPLICATION:     errStream << "Source: Application"; break;
  case GL_DEBUG_SOURCE_OTHER:           errStream << "Source: Other"; break;
  }

  errStream << '\n';

  switch (type)
  {
  case GL_DEBUG_TYPE_ERROR:               errStream << "Type: Error"; break;
  case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: errStream << "Type: Deprecated Behaviour"; break;
  case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  errStream << "Type: Undefined Behaviour"; break;
  case GL_DEBUG_TYPE_PORTABILITY:         errStream << "Type: Portability"; break;
  case GL_DEBUG_TYPE_PERFORMANCE:         errStream << "Type: Performance"; break;
  case GL_DEBUG_TYPE_MARKER:              errStream << "Type: Marker"; break;
  case GL_DEBUG_TYPE_PUSH_GROUP:          errStream << "Type: Push Group"; break;
  case GL_DEBUG_TYPE_POP_GROUP:           errStream << "Type: Pop Group"; break;
  case GL_DEBUG_TYPE_OTHER:               errStream << "Type: Other"; break;
  }

  errStream << '\n';

  switch (severity)
  {
  case GL_DEBUG_SEVERITY_HIGH:
    errStream << "Severity: high";
    break;
  case GL_DEBUG_SEVERITY_MEDIUM:
    errStream << "Severity: medium";
    break;
  case GL_DEBUG_SEVERITY_LOW:
    errStream << "Severity: low";
    break;
  case GL_DEBUG_SEVERITY_NOTIFICATION:
    errStream << "Severity: notification";
    break;
  }

  std::cout << errStream.str() << '\n';
}

struct WindowCreateInfo
{
  bool maximize{};
  bool decorate{};
  uint32_t width{};
  uint32_t height{};
};

GLFWwindow* CreateWindow(const WindowCreateInfo& createInfo)
{
  if (!glfwInit())
  {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwSetErrorCallback([](int, const char* desc)
    {
      std::cout << "GLFW error: " << desc << '\n';
    });

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_MAXIMIZED, createInfo.maximize);
  glfwWindowHint(GLFW_DECORATED, createInfo.decorate);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
  glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

  const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
  GLFWwindow* window = glfwCreateWindow(createInfo.width, createInfo.height, "ererererer", nullptr, nullptr);

  if (!window)
  {
    throw std::runtime_error("Failed to create window");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  return window;
}

void InitOpenGL()
{
  int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0)
  {
    throw std::runtime_error("Failed to initialize OpenGL");
  }
}

struct PixelInfo
{
  int x;
  int y;
  stbi_uc* pixels;
};

// Loads all the textures in the assets directory serially
void LoadTexturesSerial(CheapVector<GLuint, VEC_SIZE>& textures)
{
  std::filesystem::directory_iterator it{ "assets/" };
  for (const auto& f : it)
  {
    if (f.is_regular_file())
    {
      int x;
      int y;
      auto pixels = stbi_load(f.path().string().c_str(), &x, &y, nullptr, 4);
      if (!pixels)
      {
        throw std::exception{ "failed" };
      }

      GLuint tex;
      glCreateTextures(GL_TEXTURE_2D, 1, &tex);
      glTextureStorage2D(tex, 1, GL_RGBA8, x, y);
      glTextureSubImage2D(tex, 0, 0, 0, x, y, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      textures.emplace_back(tex);

      stbi_image_free(pixels);
    }
  }
}

// Loads all the textures in the assets directory in parallel, then uploads them serially
void LoadTexturesParallelLoadSerialUpload(CheapVector<GLuint, VEC_SIZE>& textures)
{
  CheapVector<PixelInfo, VEC_SIZE> pixels2;
  CheapVector<std::string, VEC_SIZE> filesToLoad;

  std::filesystem::directory_iterator it{ "assets/" };
  for (const auto& f : it)
  {
    if (f.is_regular_file())
    {
      filesToLoad.emplace_back(f.path().string());
    }
  }

  std::for_each(std::execution::par, filesToLoad.begin(), filesToLoad.end(),
    [&pixels2](const std::string& path)
    {
      PixelInfo pixelInfo;
      pixelInfo.pixels = stbi_load(path.c_str(), &pixelInfo.x, &pixelInfo.y, nullptr, 4);
      if (!pixelInfo.pixels)
      {
        throw std::exception{ "failed" };
      }
      pixels2.emplace_back(pixelInfo);
    });

  for (const auto& pixelInfo : pixels2)
  {
    GLuint tex;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA8, pixelInfo.x, pixelInfo.y);
    glTextureSubImage2D(tex, 0, 0, 0, pixelInfo.x, pixelInfo.y, GL_RGBA, GL_UNSIGNED_BYTE, pixelInfo.pixels);
    textures.emplace_back(tex);
    stbi_image_free(pixelInfo.pixels);
  }
}

// Loads all the textures in the assets directory in parallel, creates one mapped PBO per texture, then uploads all the textures in parallel
void LoadTexturesParallelPBO(CheapVector<GLuint, VEC_SIZE>& textures)
{
  CheapVector<PixelInfo, VEC_SIZE> pixels2;
  CheapVector<std::string, VEC_SIZE> filesToLoad;
  CheapVector<std::string, VEC_SIZE> filesMemory;

  std::filesystem::directory_iterator it{ "assets/" };
  for (const auto& f : it)
  {
    if (f.is_regular_file())
    {
      filesToLoad.emplace_back(f.path().string());
    }
  }

  std::for_each(std::execution::par, filesToLoad.begin(), filesToLoad.end(),
    [&pixels2, &filesToLoad, &filesMemory](const auto& path)
    {
      PixelInfo pixelInfo;
      auto memory = LoadFile(path, std::ios_base::binary);
      int ok = stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(memory.data()), memory.size(), &pixelInfo.x, &pixelInfo.y, nullptr);
      filesMemory.emplace_back(std::move(memory));
      pixels2.emplace_back(pixelInfo);
    });

  std::array<GLuint, VEC_SIZE> buffers;
  std::array<void*, VEC_SIZE> bufferPointers;
  glCreateBuffers(filesToLoad.size(), buffers.data());
  for (size_t i = 0; i < filesToLoad.size(); i++)
  {
    const auto& pixelInfo = pixels2[i];
    const auto& buffer = buffers[i];
    glNamedBufferStorage(buffer, pixelInfo.x * pixelInfo.y * 4, nullptr, GL_MAP_WRITE_BIT);
    bufferPointers[i] = glMapNamedBuffer(buffer, GL_WRITE_ONLY);
    assert(bufferPointers[i]);
  }

  CheapVector<size_t, VEC_SIZE> iota;
  iota.set_size(filesToLoad.size());
  std::iota(iota.begin(), iota.end(), 0);

  std::for_each(std::execution::par, iota.begin(), iota.end(),
    [&pixels2, &filesToLoad, &bufferPointers, &filesMemory](size_t i)
    {
      const auto& memory = filesMemory[i];
      PixelInfo pixelInfo;
      pixelInfo.pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(memory.data()), memory.size(), &pixelInfo.x, &pixelInfo.y, nullptr, 4);
      if (!pixelInfo.pixels)
      {
        throw std::exception{ "failed" };
      }
      std::memcpy(bufferPointers[i], pixelInfo.pixels, pixelInfo.x * pixelInfo.y * 4);
      stbi_image_free(pixelInfo.pixels);
    });

  for (size_t i = 0; i < filesToLoad.size(); i++)
  {
    const auto& pixelInfo = pixels2[i];
    GLuint tex;
    glUnmapNamedBuffer(buffers[i]);
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA8, pixelInfo.x, pixelInfo.y);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffers[i]);
    glTextureSubImage2D(tex, 0, 0, 0, pixelInfo.x, pixelInfo.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    textures.emplace_back(tex);
  }

  glDeleteBuffers(filesToLoad.size(), buffers.data());
}

auto main() -> int
{
  GLFWwindow* window = CreateWindow({ .maximize = false, .decorate = true, .width = 1280, .height = 720 });

  InitOpenGL();

  // enable debugging stuff
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(glErrorCallback, NULL);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

  int frameWidth, frameHeight;
  glfwGetFramebufferSize(window, &frameWidth, &frameHeight);

  glViewport(0, 0, frameWidth, frameHeight);
  glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
  stbi_set_flip_vertically_on_load(true);

  // convenience
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  CheapVector<GLuint, VEC_SIZE> textures;

  Timer timer;
#if METHOD == 0
  LoadTexturesSerial(textures);
  auto elapsed = timer.Elapsed_ms();
  std::cout << "Load textures serial (ms): " << elapsed << "\n";
#elif METHOD == 1
  LoadTexturesParallelLoadSerialUpload(textures);
  auto elapsed = timer.Elapsed_ms();
  std::cout << "Load textures parallel load to RAM (ms): " << elapsed << "\n";
#elif METHOD == 2
  LoadTexturesParallelPBO(textures);
  auto elapsed = timer.Elapsed_ms();
  std::cout << "Load textures parallel map (ms): " << elapsed << "\n";
#endif

  // we only need an empty vao for this
  GLuint vao;
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // a basic shader program can be bound up front
  GLuint program = CompileVertexFragmentProgram(gVertexSource, gFragmentSource);
  glUseProgram(program);

  uint64_t frame{ 0 };
  double prevFrame = glfwGetTime();
  while (!glfwWindowShouldClose(window))
  {
    double curFrame = glfwGetTime();
    double dt = curFrame - prevFrame;
    prevFrame = curFrame;
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    glClear(GL_COLOR_BUFFER_BIT);

    if (!textures.empty())
    {
      // advance texture every n seconds
      float n{ 1.0f };
      glBindTextureUnit(0, textures[int(curFrame / n) % textures.size()]);
    }
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
    frame++;
  }

  glfwTerminate();
  return 0;
}