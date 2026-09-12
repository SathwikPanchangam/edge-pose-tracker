# Distributed 6-DoF Edge Pose Tracker

An edge-optimized robotics perception pipeline built with **ROS 2 Jazzy** and **OpenCV**, designed for low-latency 6D object pose tracking with dynamic uncertainty quantification and Kalman-filtered trajectory smoothing.

[![ROS 2 Jazzy CI](https://github.com/SathwikPanchangam/edge-pose-tracker/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/SathwikPanchangam/edge-pose-tracker/actions)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Platform: Ubuntu 24.04 LTS](https://img.shields.io/badge/Platform-Ubuntu%2024.04-orange.svg)
![Hardware: Raspberry Pi 5](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red.svg)

---

## System Architecture

The pipeline decouples real-time edge perception from remote rendering using Eclipse CycloneDDS over a private wireless network:

```
[ Raspberry Pi 5 (Edge Compute) ]
  ├── camera_publisher         -> Ingests /camera/image_raw via cv_bridge
  ├── pose_estimator           -> PnP SE(3) estimation + Dynamic Covariance (IPPE)
  └── pose_filter              -> Adaptive 1D Kalman Filters for 3D trajectory smoothing
            │
      (CycloneDDS / Domain 42)
            ▼
[ Ubuntu Laptop (Remote Operator) ]
  ├── rviz2                    -> 3D TF rendering and covariance ellipsoid display
  ├── performance_evaluator    -> Real-time jitter and latency profiling
  └── service_client           -> Remote start/stop control via std_srvs/SetBool
```

---

## Key Features

* **Real-Time 6D Pose Estimation:** Solves the Perspective-n-Point problem (`cv2.solvePnP` via IPPE) to resolve translation $\mathbf{t} \in \mathbb{R}^3$ and orientation $\mathbf{q} \in S^3$.
* **Dynamic Uncertainty Modeling:** Constructs continuous $6 \times 6$ covariance matrices (`PoseWithCovarianceStamped`) dynamically scaled by reprojection residuals and quadratic depth variance ($\sigma_Z \propto Z^2$).
* **Covariance-Adaptive Kalman Filtering:** Implements discrete state-space filtering on translational axes, dynamically adjusting the Kalman gain $K$ using visual measurement variance to reject tracking jitter.
* **Full TF2 Transform Tree Integration:** Broadcasts spatial coordinate frames (`camera_optical_frame` $\to$ `target_object_filtered`) directly into the ROS 2 transform buffer.
* **Continuous Integration (CI/CD):** Automated containerized testing workflow via GitHub Actions checking algorithmic determinism and quaternion normalization.

---

## Quantitative Benchmarks

Empirical performance recorded across 100 consecutive frames on Raspberry Pi 5 (ARM64, Cortex-A76 @ 2.4 GHz):

| Metric | Raw Vision Stream | Kalman Filtered Stream | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **3D Positional Variance** | $38.4\text{ mm}^2$ | $11.9\text{ mm}^2$ | **69.0% Jitter Reduction** |
| **End-to-End Latency** | — | $11.2\text{ ms}$ (Mean) | Sub-frame processing budget |
| **Pi 5 CPU Utilization** | — | $22.4\%$ across 4 cores | $>77\%$ idle margin reserved |
| **System Memory Usage** | — | $550\text{ MB}$ total OS footprint | $<7\%$ of available 8GB RAM |
| **SoC Operating Temp** | — | $59.8^\circ\text{C}$ (Passive air) | Zero thermal throttling |

---

## Installation & Usage

### Prerequisites
* Ubuntu 24.04 LTS
* ROS 2 Jazzy Desktop
* Eclipse CycloneDDS (`ros-jazzy-rmw-cyclonedds-cpp`)

### Build Workspace
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone [https://github.com/SathwikPanchangam/edge-pose-tracker.git](https://github.com/SathwikPanchangam/edge-pose-tracker.git) edge_pose_tracker
cd ~/ros2_ws
colcon build --symlink-install --packages-select edge_pose_tracker
source install/setup.bash
```

### Launch Tracker Pipeline (Raspberry Pi 5)
```bash
ros2 launch edge_pose_tracker tracker.launch.py
```

### Remote Visualization (Operator Workstation)
```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
rviz2 -d ~/ros2_ws/src/edge_pose_tracker/config/tracker_view.rviz
```

---

## Unit Testing

Run the automated pytest verification suite locally:
```bash
colcon test --packages-select edge_pose_tracker
colcon test-result --verbose
```