# Reachy 2021 Python SDK

Reachy Python SDK is a pure python SDK library that let you control and interact with a [Reachy robot](https://www.pollen-robotics.com/reachy/). Reachy and its SDK are developed by [Pollen Robotics](https://www.pollen-robotics.com). 

This SDK is intended to work with the Reachy 2021 version.

<img src="https://www.pollen-robotics.com/img/reachy/homepage_feature.png" width="400" alt="Reachy 2021 says hello!">


It lets you get the current state of the robot (all joints position, the sensors up-to-date value, camera image) without having to think about synchronisation with the hardware.
You can also send commands to control Reachy both in joint and cartesian spaces. You can control other effectors as well such as its fan or the camera zoom.

The SDK also provides higher level function to easily create complex motions such as asynchronous goto command.

You can use the SDK over the network, either WiFi or Ethernet but you should favour the second option for low-latency control. The communication to the robot is done via [gRPC](https://grpc.io) and can thus work on most kind of network configurations. Local control of the robot (directly on Reachy's computer can simply be done using the localhost IP)

## License

The SDK, and the rest of Reachy software, is open-source and released under an Apache-2.0 License.

## Installation

The SDK can be installed on any computer running Python 3.6 or later. 

You can install the SDK:

#### From PyPi
```bash
pip install reachy-sdk
```

#### From the source

```bash
git clone https://github.com/pollen-robotics/reachy-sdk
pip install -e reachy-sdk
```

We recommend using [virtual environment](https://docs.python.org/3/tutorial/venv.html) for your development.

The SDK depends on [numpy](https://numpy.org), [opencv](https://opencv.org) and [grpc](https://grpc.io). Most of the documentation is available as [jupyter notebooks](https://jupyter.org).

## Getting Started

To get started with your Reachy installation and setup, see our [official guide](https://pollen-robotics.github.io/reachy-2021-docs/).

Connecting the SDK to a real robot is as simple as:

```python
from reachy_sdk import ReachySDK

reachy = ReachySDK(host='my-reachy-ip')
```

Gettting the current joints position can be done via:

```python
for name, joint in reachy.joints.items():
    print(f'Joint {name} is at position {joint.present_position} degree.')
```

And displaying, via matplotlib, the last image of the left camera:

```python
import cv2 as cv
from matplotlib import pyplot as plt

plt.figure()
plt.plot(cv.cvtColor(reachy.left_camera.last_frame, cv.COLOR_BGR2RGB))
```

For more advanced examples, see the [official documentation](https://pollen-robotics.github.io/reachy-2021-docs/sdk/getting-started/introduction/).

## APIs

The SDK APIs can be accessed here: [https://pollen-robotics.github.io/reachy-sdk/](https://pollen-robotics.github.io/reachy-sdk/) (generated by [Sphinx](https://www.sphinx-doc.org/en/master/)).

## Support 

This project adheres to the Contributor [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [contact@pollen-robotics.com](mailto:contact@pollen-robotics.com).

Visit [pollen-robotics.com](https://pollen-robotics.com) to learn more or join our [Dicord community](https://discord.com/invite/Kg3mZHTKgs) if you have any questions or want to share your ideas.
Follow [@PollenRobotics](https://twitter.com/pollenrobotics) on Twitter for important announcements.
