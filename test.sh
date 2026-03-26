# unload uvcvideo so it doesn't claim the camera
sudo modprobe -r uvcvideo

# verify device still listed
lsusb

# check kernel logs for errors
dmesg | tail -n 50

# run your pyrealsense check with sudo
# sudo python3 - <<'PY'
# import pyrealsense2 as rs
# ctx = rs.context()
# print("Connected devices count:", ctx.get_device_count())
# for i, d in enumerate(ctx.devices):
#     print(i, d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
# PY
# sudo python3 main.py

# from repo root in WSL
mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# then
sudo ./tools/rs-enumerate-devices