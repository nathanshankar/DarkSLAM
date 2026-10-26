/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <System.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

struct FramePaths {
    double timestamp;
    std::string ir;
    std::string depth;
};

std::string DurationText(double seconds) {
    const auto total = static_cast<long long>(std::max(0.0, seconds));
    return std::to_string(total / 60) + "m " + std::to_string(total % 60) + "s";
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: rgbd_tum_headless vocabulary settings associations output_dir [--video]\n";
        return 1;
    }
    try {
        const bool video = argc == 6 && std::string(argv[5]) == "--video";
        if (argc == 6 && !video) throw std::runtime_error("Unknown option");
        std::ifstream input(argv[3]);
        if (!input) throw std::runtime_error("Cannot open associations file");
        const auto base = std::filesystem::absolute(argv[3]).parent_path();
        std::vector<FramePaths> frames;
        std::string line;
        while (std::getline(input, line)) {
            auto start = line.find_first_not_of(" \t\r");
            if (start == std::string::npos || line[start] == '#') continue;
            FramePaths frame;
            double depth_stamp;
            std::istringstream row(line);
            if (!(row >> frame.timestamp >> frame.ir >> depth_stamp >> frame.depth))
                throw std::runtime_error("Malformed association: " + line);
            if (!std::isfinite(frame.timestamp) || !std::isfinite(depth_stamp))
                throw std::runtime_error("Non-finite timestamp");
            if (!frames.empty() && frame.timestamp <= frames.back().timestamp)
                throw std::runtime_error("IR timestamps must be strictly increasing");
            frame.ir = (base / frame.ir).string();
            frame.depth = (base / frame.depth).string();
            frames.push_back(frame);
        }
        if (frames.empty()) throw std::runtime_error("No associated frames");
        // Validate all images before starting worker threads.
        for (const auto& frame : frames) {
            const auto ir = cv::imread(frame.ir, cv::IMREAD_UNCHANGED);
            const auto depth = cv::imread(frame.depth, cv::IMREAD_UNCHANGED);
            if (ir.type() != CV_8UC1 || ir.size() != cv::Size(640, 480) ||
                depth.type() != CV_16UC1 || depth.size() != ir.size())
                throw std::runtime_error("Invalid IR/depth image pair: " + frame.ir);
        }
        const std::filesystem::path output = std::filesystem::absolute(argv[4]);
        std::filesystem::create_directories(output);
        std::ofstream timings(output / "tracking.csv");
        if (!timings) throw std::runtime_error("Cannot open tracking.csv");
        timings << "timestamp,tracking_seconds,state" << std::endl;
        timings << std::fixed << std::setprecision(9);
        ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);
        cv::VideoWriter writer;
        const auto sequenceBegin = std::chrono::steady_clock::now();
        size_t completed = 0;
        try {
            for (const auto& frame : frames) {
                auto ir = cv::imread(frame.ir, cv::IMREAD_UNCHANGED);
                auto depth = cv::imread(frame.depth, cv::IMREAD_UNCHANGED);
                if (ir.empty() || depth.empty()) throw std::runtime_error("Image disappeared during evaluation");
                const auto begin = std::chrono::steady_clock::now();
                slam.TrackRGBD(ir, depth, frame.timestamp);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                const int state = slam.GetTrackingState();
                // A complete row must be visible immediately to live monitors.
                timings << frame.timestamp << ',' << elapsed << ',' << state << std::endl;
                if (!timings) throw std::runtime_error("Could not write tracking.csv");
                ++completed;
                if (completed == 1 || completed % 10 == 0 || completed == frames.size()) {
                    const double wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - sequenceBegin).count();
                    const double remaining = wall / completed * (frames.size() - completed);
                    std::cout << "IR-D progress: " << completed << '/' << frames.size()
                              << " (" << std::fixed << std::setprecision(1)
                              << 100.0 * completed / frames.size() << "%), state=" << state
                              << ", elapsed=" << DurationText(wall)
                              << ", ETA=" << DurationText(remaining) << std::endl;
                }
                if (video) {
                    auto view = slam.mpFrameDrawer->DrawFrame();
                    if (!writer.isOpened()) {
                        writer.open((output / "tracking.mp4").string(),
                                    cv::VideoWriter::fourcc('m','p','4','v'), 30, view.size());
                        if (!writer.isOpened()) throw std::runtime_error("Could not open tracking video");
                    }
                    writer.write(view);
                }
            }
        } catch (...) {
            slam.Shutdown();
            throw;
        }
        std::cout << "IR-D progress: tracking finished; waiting for mapping and exporting trajectories"
                  << std::endl;
        slam.Shutdown();
        slam.SaveTrajectoryTUM((output / "trajectory.txt").string());
        slam.SaveKeyFrameTrajectoryTUM((output / "keyframes.txt").string());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "IR-D evaluation failed: " << error.what() << std::endl;
        return 1;
    }
}
