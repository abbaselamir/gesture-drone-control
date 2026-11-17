import json
import socket
import math
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

motors = [
    robot.getDevice("front left propeller"),
    robot.getDevice("front right propeller"),
    robot.getDevice("rear left propeller"),
    robot.getDevice("rear right propeller")
]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))
sock.setblocking(False)

K_VERTICAL_THRUST = 68.5
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
K_ROLL_COMMAND = 5.0
K_PITCH_COMMAND = 3.0

TAKEOFF_ALTITUDE = 1.5
target_altitude = 1.5
previous_mode = "LANDED"
cmd = {"mode": "LANDED", "vx": 0.0, "vy": 0.0, "vz": 0.0}

while robot.step(timestep) != -1:
    try:
        data, _ = sock.recvfrom(1024)
        cmd = json.loads(data.decode())
    except BlockingIOError:
        pass

    roll, pitch, yaw = imu.getRollPitchYaw()
    x, y, altitude = gps.getValues()
    roll_vel, pitch_vel, yaw_vel = gyro.getValues()

    if cmd["mode"] == "LANDED":
        previous_mode = "LANDED"
        target_altitude = max(0.05, target_altitude - 0.01)
        if altitude < 0.12:
            for m in motors:
                m.setVelocity(0.0)
            continue
    else:
        if previous_mode == "LANDED":
            target_altitude = TAKEOFF_ALTITUDE
        previous_mode = "FLYING"
        target_altitude = max(0.3, min(10.0, target_altitude + cmd["vz"] * (timestep / 1000.0)))

    roll_disturbance = -cmd["vy"] * K_ROLL_COMMAND
    pitch_disturbance = -cmd["vx"] * K_PITCH_COMMAND

    roll_input = K_ROLL_P * max(-1.0, min(1.0, roll)) + roll_vel + roll_disturbance
    pitch_input = K_PITCH_P * max(-1.0, min(1.0, pitch)) + pitch_vel + pitch_disturbance
    clamped_diff_alt = max(-1.0, min(1.0, target_altitude - altitude + 0.04))
    vertical_input = K_VERTICAL_P * math.pow(clamped_diff_alt, 3.0)

    motors[0].setVelocity(K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input)
    motors[1].setVelocity(-(K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input))
    motors[2].setVelocity(-(K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input))
    motors[3].setVelocity(K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input)
