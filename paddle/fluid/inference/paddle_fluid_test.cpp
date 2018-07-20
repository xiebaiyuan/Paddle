//
// Created by liuRuiLong on 2018/4/23.
//

#include <map>
#include <random>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_int32(batch_size,
1, "Batch size of input data");
DEFINE_string(dirname,
"/Users/xiebaiyuan/PaddleProject/Paddle/paddle/fluid/inference/tests/image_classification_resnet.inference.model", "Directory of the inference model.");
DEFINE_string(combine_dirname,
"/Users/xiebaiyuan/PaddleProject/Paddle/paddle/fluid/inference/tests/googlenet_combine", "combine dir");
DEFINE_int32(repeat,
1, "Running the inference program repeat times");

template<typename T>
void SetupTensor(paddle::framework::LoDTensor *input,
                 paddle::framework::DDim dims, T lower, T upper) {
    static unsigned int seed = 100;
    std::mt19937 rng(seed++);
    std::uniform_real_distribution<double> uniform_dist(0, 1);

    T *input_ptr = input->mutable_data<T>(dims, paddle::platform::CPUPlace());

    for (int i = 0; i < input->numel(); ++i) {
        input_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
}

template<typename T>
void SetupTensor(paddle::framework::LoDTensor *input,
                 paddle::framework::DDim dims, const std::vector<T> &data) {
    CHECK_EQ(paddle::framework::product(dims), static_cast<int64_t>(data.size()));
    T *input_ptr = input->mutable_data<T>(dims, paddle::platform::CPUPlace());
    memcpy(input_ptr, data.data(), input->numel() * sizeof(T));
}

template<typename T>
void SetupLoDTensor(paddle::framework::LoDTensor *input,
                    const paddle::framework::LoD &lod, T lower, T upper) {
    input->set_lod(lod);
    int dim = lod[0][lod[0].size() - 1];
    SetupTensor<T>(input, {dim, 1}, lower, upper);
}

template<typename T>
void SetupLoDTensor(paddle::framework::LoDTensor *input,
                    paddle::framework::DDim dims,
                    const paddle::framework::LoD lod,
                    const std::vector<T> &data) {
    const size_t level = lod.size() - 1;
    CHECK_EQ(dims[0], static_cast<int64_t>((lod[level]).back()));
    input->set_lod(lod);
    SetupTensor<T>(input, dims, data);
}

template<typename T>
void CheckError(const paddle::framework::LoDTensor &output1,
                const paddle::framework::LoDTensor &output2) {
    // Check lod information

    T err = static_cast<T>(0);
    if (typeid(T) == typeid(float)) {
        err = 1E-3;
    } else if (typeid(T) == typeid(double)) {
        err = 1E-6;
    } else {
        err = 0;
    }

    size_t count = 0;
    for (int64_t i = 0; i < output1.numel(); ++i) {
        if (fabs(output1.data<T>()[i] - output2.data<T>()[i]) > err) {
            count++;
        }
    }
}

template<typename Place, bool CreateVars = true, bool PrepareContext = false, bool CreateLocalScope = false>
void TestInference(const std::string &dirname,
                   const std::vector<paddle::framework::LoDTensor *> &cpu_feeds,
                   const std::vector<paddle::framework::LoDTensor *> &cpu_fetchs,
                   const int repeat = 1, const bool is_combined = false) {
    // 1. Define place, executor, scope
    auto place = Place();
    auto executor = paddle::framework::Executor(place);
    auto *scope = new paddle::framework::Scope();

    // Profile the performance
    paddle::platform::ProfilerState state;
    if (paddle::platform::is_cpu_place(place)) {
        state = paddle::platform::ProfilerState::kCPU;
    } else {
#ifdef PADDLE_WITH_CUDA
        state = paddle::platform::ProfilerState::kCUDA;
        // The default device_id of paddle::platform::CUDAPlace is 0.
        // Users can get the device_id using:
        //   int device_id = place.GetDeviceId();
        paddle::platform::SetDeviceId(0);
#else
        PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
    }

    // 2. Initialize the inference_program and load parameters
    std::unique_ptr<paddle::framework::ProgramDesc> inference_program;

    // Enable the profiler
    paddle::platform::EnableProfiler(state);
    {
        paddle::platform::RecordEvent record_event(
                "init_program",
                paddle::platform::DeviceContextPool::Instance().Get(place));

        if (is_combined) {
            // All parameters are saved in a single file.
            // Hard-coding the file names of program and parameters in unittest.
            // The file names should be consistent with that used in Python API
            //  `fluid.io.save_inference_model`.
            std::string prog_filename = "model";
            std::string param_filename = "params";
            inference_program = paddle::inference::Load(
                    &executor, scope, dirname + "/" + prog_filename,
                    dirname + "/" + param_filename);
        } else {
            // Parameters are saved in separate files sited in the specified
            // `dirname`.
            inference_program = paddle::inference::Load(&executor, scope, dirname);
        }
    }
    // Disable the profiler and print the timing information
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kDefault,
                                      "load_program_profiler.txt");
    paddle::platform::ResetProfiler();

    // 3. Get the feed_target_names and fetch_target_names
    const std::vector<std::string> &feed_target_names =
            inference_program.get()->GetFeedTargetNames();
    const std::vector<std::string> &fetch_target_names =
            inference_program.get()->GetFetchTargetNames();

    // 4. Prepare inputs: set up maps for feed targets
    std::map<std::string, const paddle::framework::LoDTensor *> *feed_targets = new std::map<std::string, const paddle::framework::LoDTensor *>;
    for (size_t i = 0; i < feed_target_names.size(); ++i) {
        // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
        (*feed_targets)[feed_target_names[i]] = cpu_feeds[i];
    }

    // 5. Define Tensor to get the outputs: set up maps for fetch targets
    std::map<std::string, paddle::framework::LoDTensor *> *fetch_targets = new std::map<std::string, paddle::framework::LoDTensor *>;
    for (size_t i = 0; i < fetch_target_names.size(); ++i) {
        (*fetch_targets)[fetch_target_names[i]] = cpu_fetchs[i];
    }

    // 6. Run the inference program
    {
        if (!CreateVars) {
            // If users don't want to create and destroy variables every time they
            // run, they need to set `create_vars` to false and manually call
            // `CreateVariables` before running.
            executor.CreateVariables(*inference_program, scope, 0);
        }

        // Ignore the profiling results of the first run
        std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx;
        if (PrepareContext) {
            ctx = executor.Prepare(*inference_program, 0);
            executor.RunPreparedContext(ctx.get(), scope, feed_targets, fetch_targets, CreateLocalScope, CreateVars);
        } else {
            executor.Run(*inference_program, scope, feed_targets, fetch_targets, CreateLocalScope, CreateVars);
        }

        // Enable the profiler
        paddle::platform::EnableProfiler(state);

        // Run repeat times to profile the performance
        for (int i = 0; i < repeat; ++i) {
            paddle::platform::RecordEvent record_event(
                    "run_inference",
                    paddle::platform::DeviceContextPool::Instance().Get(place));

            if (PrepareContext) {
                // Note: if you change the inference_program, you need to call
                // executor.Prepare() again to get a new ExecutorPrepareContext.
                executor.RunPreparedContext(ctx.get(), scope, feed_targets,
                                            fetch_targets, CreateVars);
            } else {
                executor.Run(*inference_program, scope, feed_targets, fetch_targets,
                             CreateVars);
            }
        }

        // Disable the profiler and print the timing information
        paddle::platform::DisableProfiler(
                paddle::platform::EventSortingKey::kDefault,
                "run_inference_profiler.txt");
        paddle::platform::ResetProfiler();
    }

    delete scope;
    delete feed_targets;
    delete fetch_targets;
}


int main() {
    paddle::inference::Init({});

    if (FLAGS_dirname.empty() || FLAGS_batch_size < 1 || FLAGS_repeat < 1) {
        LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model "
                "--batch_size=1 --repeat=1";
    }

    LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
    std::string dirname = FLAGS_dirname;

    // 0. Call `paddle::framework::InitDevices()` initialize all the devices
    // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

    paddle::framework::LoDTensor input;
    // Use normilized image pixels as input data,
    // which should be in the range [0.0, 1.0].
    SetupTensor<float>(&input, {FLAGS_batch_size, 3, 224, 224},
                       static_cast<float>(0), static_cast<float>(1));
    std::vector<paddle::framework::LoDTensor *> cpu_feeds;
    cpu_feeds.push_back(&input);

    paddle::framework::LoDTensor output1;
    std::vector<paddle::framework::LoDTensor *> cpu_fetchs1;
    cpu_fetchs1.push_back(&output1);

    // Run inference on CPU
    LOG(INFO) << "--- CPU Runs: ---";
    TestInference<paddle::platform::CPUPlace, false, true>(
            FLAGS_combine_dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat, true);
    LOG(INFO) << output1.dims();

    return 0;
}
