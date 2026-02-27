#include "SuperPointExtractor.h"
#include <iostream>

namespace ORB_SLAM3 {

SuperPointExtractor::SuperPointExtractor(const std::string& model_path,
                                       int nfeatures,
                                       float scaleFactor,
                                       int nlevels,
                                       float confidence_threshold)
    : mConfidenceThreshold(confidence_threshold),
      mnFeatures(nfeatures),
      mfScaleFactor(scaleFactor),
      mnLevels(nlevels),
      mDevice(torch::kCPU)
{
    // Decide device first
    mDevice = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                          : torch::Device(torch::kCPU);

    try {
        // Load the TorchScript model directly onto that device
        std::cout << "Loading SuperPoint model from: " << model_path
                  << " on device: " << (mDevice.is_cuda() ? "CUDA" : "CPU") << std::endl;

        mModel = torch::jit::load("/home/c12888mg/Code/DarkSLAM/SUPERSLAM3/Weights/SuperPointNet.pt", mDevice);
        mModel.to(mDevice);
        mModel.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading SuperPoint model: " << e.what() << std::endl;
        throw;
    }

    // Compute scale pyramid parameters
    mvScaleFactors.resize(mnLevels);
    mvLevelSigma2.resize(mnLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for(int i=1; i<mnLevels; i++)
    {
        mvScaleFactors[i] = mvScaleFactors[i-1]*mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i]*mvScaleFactors[i];
    }

    mvInvScaleFactors.resize(mnLevels);
    mvInvLevelSigma2.resize(mnLevels);
    for(int i=0; i<mnLevels; i++)
    {
        mvInvScaleFactors[i] = 1.0f/mvScaleFactors[i];
        mvInvLevelSigma2[i] = 1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(mnLevels);

    isLeft = true;
}

int SuperPointExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::OutputArray _descriptors,
                                  std::vector<int>& vLappingArea)
{
    cv::Mat image = _image.getMat();
    cv::Mat vis_img = image.clone();
    if(image.empty())
        return -1;

    // Create grayscale image if necessary
    cv::Mat grayImage;
    if(image.channels() > 1)
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    else
        grayImage = image.clone();

    ComputePyramid(grayImage);

    for(int level = 0; level < mnLevels; ++level)
    {
        torch::Tensor inputTensor = PreprocessImage(mvImagePyramid[level]);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);

        auto output = mModel.forward(inputs).toTuple();
        auto scores     = output->elements()[0].toTensor().to(torch::kCPU);
        auto dense_desc = output->elements()[1].toTensor().to(torch::kCPU);

        // Extract initial keypoints based on confidence threshold
        std::vector<cv::KeyPoint> level_keypoints = ExtractKeypoints(scores, mConfidenceThreshold, level);

        if (level_keypoints.empty())
            continue;


        ApplyANMS(level_keypoints, mnFeatures/mnLevels);
        cv::Mat level_descriptors = ExtractDescriptors(dense_desc, level_keypoints);
            const float scale = mvScaleFactors[level];
        for (auto &kp : level_keypoints)
        {
            kp.pt *= scale;
        }
        // Append to overall keypoints and descriptors
        keypoints.insert(keypoints.end(), level_keypoints.begin(), level_keypoints.end());
        if(level == 0)
        {
            _descriptors.create(level_descriptors.rows, level_descriptors.cols, level_descriptors.type());
            level_descriptors.copyTo(_descriptors);
        }
        else
        {
            cv::Mat existing_desc = _descriptors.getMat();
            cv::Mat combined_desc;
            cv::vconcat(existing_desc, level_descriptors, combined_desc);
            _descriptors.create(combined_desc.rows, combined_desc.cols, combined_desc.type());
            combined_desc.copyTo(_descriptors);
        }

    }

    // std::cout << "SuperPoint detected " << keypoints.size() << " keypoints" << std::endl;
    // std::cout << "Confidence threshold: " << mConfidenceThreshold << std::endl;

    // // For monocular visualisation
    // //Visualise keypoints
    // cv::drawKeypoints(vis_img, keypoints, vis_img, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("Vis", vis_img);
    // cv::waitKey(0);

    // Visualisation for first left image only for stereo
    // if(isLeft){
    //     //Visualise keypoints
    //     cv::drawKeypoints(vis_img, keypoints, vis_img, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DEFAULT);
            
    //     cv::imshow("Vis", vis_img);
    //     cv::waitKey(0);
    //     isLeft = false;
    // }
    // else{
    //     isLeft = true;
    // }

    // std::cout << "[SuperPoint] Total keypoints after ANMS: " << keypoints.size() << std::endl;

    return keypoints.size();
}

torch::Tensor SuperPointExtractor::PreprocessImage(const cv::Mat& image) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0/255.0);

    // 1) Create CPU tensor from image data
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor tensor_img = torch::from_blob(
        float_img.data,
        {1, 1, float_img.rows, float_img.cols},
        options
    ).clone();   // clone so tensor owns its memory

    // 2) Move to the same device as the model (GPU if mDevice is CUDA)
    tensor_img = tensor_img.to(mDevice);

    return tensor_img;
}

std::vector<cv::KeyPoint> SuperPointExtractor::ExtractKeypoints(const torch::Tensor& semi,
                                                                float threshold, int level)
{
    // semi: [1, 65, Hc, Wc] from SuperPointNet.pt
    // We will produce full-res heatmap [1,1,H,W] with H = Hc*8, W = Wc*8

    // Move to CPU for easier access
    torch::Tensor s = semi.to(torch::kCPU);

    // Remove batch dim: [65, Hc, Wc]
    s = s.squeeze(0);  // [65, Hc, Wc]

    const int C  = s.size(0);   // 65
    const int Hc = s.size(1);
    const int Wc = s.size(2);

    // 1) Softmax over channel dim (65)
    s = torch::softmax(s, /*dim=*/0);   // [65, Hc, Wc]

    // 2) Drop dustbin channel → [64, Hc, Wc]
    torch::Tensor nodust = s.slice(/*dim=*/0, /*start=*/0, /*end=*/64);  // [64, Hc, Wc]

    // 3) Reshape to [8, 8, Hc, Wc]
    nodust = nodust.view({8, 8, Hc, Wc});   // [8,8,Hc,Wc]

    // 4) Rearrange to full-res [1,1,Hc*8,Wc*8]
    //    (Hc,8,Wc,8) → (Hc*8, Wc*8)
    nodust = nodust.permute({2, 0, 3, 1});            // [Hc,8,Wc,8]
    nodust = nodust.contiguous().view({1, 1, Hc*8, Wc*8}); // [1,1,H,W]

    torch::Tensor heatmap = nodust;  // [1,1,H,W]
    const int H = heatmap.size(2);
    const int W = heatmap.size(3);

    // 5) Simple 3x3 NMS via max-pooling
    torch::Tensor max_pool = torch::max_pool2d(heatmap, {3,3}, {1,1}, {1,1});
    torch::Tensor keep = (heatmap == max_pool) & (heatmap > threshold); // Bool tensor [1,1,H,W]

    // Squeeze to [H,W] and keep as Bool
    // torch::Tensor keep2d = keep.squeeze(0).squeeze(0);       // [H,W], dtype: Bool
    // torch::Tensor heat2d = heatmap.squeeze(0).squeeze(0);    // [H,W], dtype: Float
    torch::Tensor keep2d = keep.squeeze(0).squeeze(0).contiguous();
    torch::Tensor heat2d = heatmap.squeeze(0).squeeze(0).contiguous();


    auto keep_acc = keep2d.accessor<bool, 2>();
    auto heat_acc = heat2d.accessor<float, 2>();

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(1000);

    float scale = mvScaleFactors[level];

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (keep_acc[y][x])
            {
                float score = heat_acc[y][x];

                cv::KeyPoint kp(cv::Point2f((float)x, (float)y),
                8.0f, -1.f, score);
                // if(level != 0)
                //     kp.pt *= scale;
                kp.octave = level;
                keypoints.emplace_back(kp);

            }
        }
    }

    // std::cout << "[SuperPoint] Raw detected keypoints (full-res) = "
    //           << keypoints.size() << std::endl;

    return keypoints;
}

cv::Mat SuperPointExtractor::ExtractDescriptors(
        const torch::Tensor& descriptors,
        const std::vector<cv::KeyPoint>& keypoints)
{
    // using bilinear interpolation for upsampling the descriptors
    // descriptors: [1, C, Hc, Wc] (coarse descriptor map)
    torch::Tensor desc_map = descriptors.to(torch::kCPU).contiguous();

    const int C  = desc_map.size(1);  // descriptor dimension, e.g. 256
    const int Hc = desc_map.size(2);  // descriptor map height
    const int Wc = desc_map.size(3);  // descriptor map width

    const float DOWNSAMPLE = 8.0f;    // SuperPoint stride between heatmap and descriptor map

    cv::Mat desc(static_cast<int>(keypoints.size()), C, CV_32F);

    auto acc = desc_map.accessor<float, 4>(); // [1, C, Hc, Wc]

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const cv::KeyPoint& kp = keypoints[i];

        // Map from image coordinates → descriptor map coordinates
        float xf = kp.pt.x / DOWNSAMPLE;
        float yf = kp.pt.y / DOWNSAMPLE;

        // Clamp inside valid range
        xf = std::max(0.0f, std::min(xf, static_cast<float>(Wc - 1)));
        yf = std::max(0.0f, std::min(yf, static_cast<float>(Hc - 1)));

        int x0 = static_cast<int>(std::floor(xf));
        int y0 = static_cast<int>(std::floor(yf));
        int x1 = std::min(x0 + 1, Wc - 1);
        int y1 = std::min(y0 + 1, Hc - 1);

        float dx = xf - static_cast<float>(x0);
        float dy = yf - static_cast<float>(y0);

        float w00 = (1.0f - dx) * (1.0f - dy);
        float w10 = dx * (1.0f - dy);
        float w01 = (1.0f - dx) * dy;
        float w11 = dx * dy;

        float* row = desc.ptr<float>(static_cast<int>(i));

        // Bilinear interpolation over the 4 neighbors
        for (int d = 0; d < C; ++d)
        {
            float v00 = acc[0][d][y0][x0];
            float v10 = acc[0][d][y0][x1];
            float v01 = acc[0][d][y1][x0];
            float v11 = acc[0][d][y1][x1];

            row[d] = w00 * v00 +
                     w10 * v10 +
                     w01 * v01 +
                     w11 * v11;
        }

        // L2-normalize this descriptor (strongly recommended for SuperPoint)
        float norm = 0.0f;
        for (int d = 0; d < C; ++d)
            norm += row[d] * row[d];

        norm = std::sqrt(norm) + 1e-8f; // avoid division by zero

        for (int d = 0; d < C; ++d)
            row[d] /= norm;
    }

    return desc;
}

void SuperPointExtractor::ApplyANMS(std::vector<cv::KeyPoint>& keypoints,
                                    const int num_features)
{
    if (static_cast<int>(keypoints.size()) <= num_features){
        // std::cout << "Not enough features: " << keypoints.size() << "keypoints, min features " << num_features << std::endl;
        return;
    }

    // 1. Sort keypoints by response (confidence), descending
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    const size_t N = keypoints.size();
    std::vector<float> radii(N, std::numeric_limits<float>::max());

    // 2. For each keypoint i, compute distance to the nearest higher-score keypoint j < i
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float dist = dx * dx + dy * dy;

            // Only update radius of the *lower-score* point (i)
            if (dist < radii[i])
                radii[i] = dist;
        }
    }

    // 3. Sort indices by radius, descending (larger radius = more isolated)
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&radii](size_t i1, size_t i2) {
                  return radii[i1] > radii[i2];
              });

    // 4. Keep top num_features keypoints
    std::vector<cv::KeyPoint> new_keypoints;
    new_keypoints.reserve(num_features);

    const int K = std::min(num_features, static_cast<int>(N));
    for (int k = 0; k < K; ++k)
    {
        new_keypoints.push_back(keypoints[indices[k]]);
    }

    keypoints = std::move(new_keypoints);
    // std::cout << "Num features th: " << num_features << std::endl;
}


void SuperPointExtractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < mnLevels; ++level)
    {
        float scale = mvInvScaleFactors[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::resize(image, mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
    }
}

} // namespace ORB_SLAM3