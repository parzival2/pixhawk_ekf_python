## EKF in python

This is an example project for reference that I did before moving on to 
the C++ implementation in Pixhawk. For the pixhawk implementation please visit
[pixhawk_ekf](https://github.com/parzival2/pixhawk_ekf) repo where the same
code will be implemented in C++. For this code to work you also need to
visit that repository and build and flash the code from that repo.

Uses ros_serial to get `imu` messages on `imu\data_raw` topic as the [imu_tools](http://wiki.ros.org/imu_filter_madgwick) will listen on that topic. The imu messages are published
at 100hz using polling and the interrupt based approach will be implemented in the original
C++ repo.

Visualization cube and pose messages are visualized using rviz `markers` and `PoseStamped` messages.

![rviz visualization](images/pixhawkekf_rviz_visualization.gif)

## TODOs

- [x] Use gyro bias in EKF state
- [ ] Try to use magnetometer in pixhawk
- [ ] Seperate ros and EKF implementation
- [ ] Calibrate Accelerometer and if(magentometer)

### Resources

Quaternion derivative
<https://math.stackexchange.com/questions/1896379/how-to-use-the-quaternion-derivative>

Manifold toolkit
<https://arxiv.org/pdf/1107.1119.pdf>

Quaternion kinematics
<http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf>

Fundermentals of Small unmanned aircraft flight 
<https://www.jhuapl.edu/Content/techdigest/pdf/V31-N02/31-02-Barton.pdf>