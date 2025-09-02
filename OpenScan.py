basepath = '/home/pi/OpenScan/'
from os.path import isfile
import os
import cv2
import numpy as np
from skimage import feature, color, transform, measure
from PIL import Image
import glob

def stack_images():
    images = []
    file_list = sorted(glob.glob('/home/pi/OpenScan/tmp2/stacking/stack*.jpg'))

    for file in file_list:
        image = cv2.imread(file,1)
        images.append(image)

    img_sum = np.zeros(image.shape)
    img_cnt = np.zeros(image.shape)

    for img in images:
        no_sat = np.any(img < 255, 2)
        no_sat = np.dstack((no_sat, no_sat, no_sat))
        img_cnt = img_cnt + no_sat.astype(float)
        img[np.logical_not(no_sat)] = 0
        img_sum = img_sum + img.astype(float)

    nonzeros_img_cnt = np.maximum(img_cnt, 1)
    avg_img = img_sum / nonzeros_img_cnt
    avg_img[img_cnt == 0] = 255
    avg_img = np.round(avg_img).astype(np.uint8)
    cv2.imwrite('/home/pi/OpenScan/tmp2/results.jpg', avg_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

def stack_imagesOLD():
    orb = cv2.ORB_create()
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None

    file_list = sorted(glob.glob('/home/pi/OpenScan/tmp2/stack*.jpg'))
    for file in file_list:
        image = cv2.imread(file,1)
        imageF = image.astype(np.float32) / 255
        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)
        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    cv2.imwrite('/home/pi/OpenScan/tmp2/stacking/results.jpg', stacked_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return

def get_centroid():
    downscale = 2
    image = cv2.imread('/home/pi/OpenScan/tmp2/preview.jpg', cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image_width = float(image.shape[0])
    image_height = float(image.shape[1])

    image = transform.resize(image, (image_width // downscale, image_height // downscale), anti_aliasing=True)

    try:
        orb = feature.ORB(n_keypoints=10000, downscale=1.2, fast_n=2, fast_threshold=0.2 , n_scales=3, harris_k=0.001)
        orb.detect_and_extract(image)
        keypoints = orb.keypoints
    except:
        return np.array([0, 0, 0, 0])

    data = list(tuple(i) for i in keypoints)
    y, x = zip(*data)
    l = len(x)

    centroid = np.array([4656.0, 3496.0, image_height*0.1, image_width*0.1])
    return centroid

def load_bool(name):
    filename = basepath+'settings/'+name
    if not isfile(filename):
        return
    with open(filename, 'r') as file:
        value = file.read().replace('\n','')
    if value == '1' or value == 'True' or value =='true':
        value = True
    else:
        value = False
    return value

def fade_led(pin_led, fade_steps, duty_max, dir = True):
    import RPi.GPIO as GPIO
    import time
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(pin_led, GPIO.OUT)
    pwm = GPIO.PWM(pin_led, 200)

    if dir:
        pwm.start(0)
        for duty_cycle in range(0, fade_steps*10, 1):  # Increase duty cycle in steps
            pwm.ChangeDutyCycle(duty_max*duty_cycle/(10*fade_steps))
            time.sleep(0.001)  # Pause between steps (adjust as needed)
    else:
        pwm.start(duty_max)
        for duty_cycle in range(fade_steps*10,0, -1):  # Increase duty cycle in steps
            pwm.ChangeDutyCycle(duty_max*duty_cycle/(10*fade_steps))
            time.sleep(0.001)  # Pause between steps (adjust as needed)
    pwm.stop()

def check_hotspot_mode(interface="wlan0"):
    import subprocess
    try:
        output = subprocess.check_output(["iwconfig", interface]).decode("utf-8")
        if "Mode:Master" in output:
            return True
        elif "Mode:Managed" in output:
            return False
        else:
            return False
    except subprocess.CalledProcessError as e:
        return False

def add_wifi_network(ssid, password, country):
    import re
    conf_file = "/etc/wpa_supplicant/wpa_supplicant-wlan0.conf"

    if not os.path.exists(conf_file):
        return False

    if not (ssid and password and country):
        return False

    with open(conf_file, "r") as f:
        content = f.read()

    updated_content = re.sub(r'country=\w+', f'country={country}', content)

    if f'ssid="{ssid}"' in content:
        network_block_pattern = re.compile(
            r'network=\{\s*ssid="' + re.escape(ssid) + r'".*?psk=".*?".*?\}', re.DOTALL
        )
        updated_network_block = f'network={{\n    ssid="{ssid}"\n    psk="{password}"\n    key_mgmt=WPA-PSK\n}}'
        updated_content = network_block_pattern.sub(updated_network_block, updated_content)
    else:
        network_block = f'\nnetwork={{\n    ssid="{ssid}"\n    psk="{password}"\n    key_mgmt=WPA-PSK\n}}\n'
        updated_content += network_block

    with open(conf_file, "w") as f:
        f.write(updated_content)
    os.system("sudo systemctl restart wpa_supplicant@wlan0")
    return True

def load_str(name):
    filename = basepath+'settings/'+name
    if not isfile(filename):
        return
    with open(filename, 'r') as file:
        value = file.read().replace('\n','')
    return value

def load_int(name):
    filename = basepath+'settings/'+name
    if not isfile(filename):
        return
    with open(filename, 'r') as file:
        value = int(file.read().replace('\n',''))
    return value

def load_float(name):
    filename = basepath+'settings/'+name
    if not isfile(filename):
        return
    with open(filename, 'r') as file:
        value = float(file.read().replace('\n',''))
    return value

def save(name, value):
    filename = basepath+'settings/'+name
    with open(filename, 'w+') as file:
        file.write(str(value))
    return

def OpenScanCloud(cmd, msg):
    from requests import get
    osc_user = 'openscan'
    osc_pw = 'free'
    osc_server = 'http://openscanfeedback.dnsuser.de:1334/'

    try:
        r = get(osc_server + cmd, auth=(osc_user, osc_pw), params=msg)
    except:
        r = type('obj', (object,), {'status_code' : 404, 'text':None})
    return r

def camera(cmd, msg = {}):
    from requests import get
    flask = 'http://127.0.0.1:1312/'
    try:
        r = get(flask + cmd, params=msg)
        data = r.json()
        return data    #status_code
    except:
        return 400

def motorrun(motor,angle,endstop=False):
    #motor can be "rotor", "tt" or "extra"
    import RPi.GPIO as GPIO
    from time import sleep
    from math import cos
    msg = {'cmd':'set'}

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    spr = load_int(motor + '_stepsperrotation')
    dirpin = load_int('pin_' + motor + '_dir')
    steppin = load_int('pin_' + motor +'_step')
    dir = load_int(motor + '_dir')
    ramp = load_int(motor + '_accramp')
    acc = load_float(motor + '_acc')
    delay_init = load_float(motor + '_delay')
    delay = delay_init

    step_count=int(angle*spr/360) * dir
    GPIO.setup(dirpin, GPIO.OUT)
    GPIO.setup(steppin, GPIO.OUT)

    if endstop:
        endstop_pin = load_int('pin_' + motor + '_endstop')
        endstop_pushed = load_bool(motor + '_endstop_pushed')
        GPIO.setup(endstop_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)

    if (step_count>0):
        GPIO.output(dirpin, GPIO.HIGH)
    if(step_count<0):
        GPIO.output(dirpin, GPIO.LOW)
        step_count=-step_count
    for x in range(step_count):
        if endstop:
            # Stop movement if endstop is pushed AND if rotor is moving and isn't going away from the endstop. 
            if GPIO.input(endstop_pin) == endstop_pushed and (motor == 'rotor' and GPIO.input(dirpin) == False):
                i = 0
                while i <= 10:
                    if GPIO.input(endstop_pin) != endstop_pushed:
                        i = 11
                    if i == 10:
                        return
                    i = i + 1

        GPIO.output(steppin, GPIO.HIGH)
        if x<=ramp and x<=step_count/2:
            delay = delay_init * (1 + -1/acc*cos(1*(ramp-x)/ramp)+1/acc)
            #delay=delay_init+(ramp-x)*(delay_init)/acc
        elif step_count-x<=ramp and x>step_count/2:
            delay = delay_init * (1-1/acc*cos(1*(ramp+x-step_count)/ramp)+1/acc)
            #delay=delay_init+(ramp-step_count+x)*(delay_init)/acc
        else:
            delay = delay_init
        sleep(delay)
        GPIO.output(steppin, GPIO.LOW)
        sleep(delay)

def ringlight(number,state):
    import RPi.GPIO as GPIO
    msg = {'cmd':'set'}
    pin = load_int('pin_ringlight' + str(number))
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, state)

def take_photo(file):
    from os import system
    filepath = basepath + file

    model=load_str('model')

    shutter = str(load_int('cam_shutter'))
    saturation = load_str('cam_saturation')
    contrast = load_str('cam_contrast')
    awbg_red = load_str('cam_awbg_red')
    awbg_blue = load_str('cam_awbg_blue')
    gain = load_str('cam_gain')
    quality = load_int('cam_jpeg_quality')
    filepath2 = '/home/pi/OpenScan/tmp/tmp.jpg'
    #width = load_str('cam_resx')
    #height = load_str('cam_resy')
    timeout = load_str('cam_timeout')
    cropx = load_int('cam_cropx')/200
    cropy = load_int('cam_cropy')/200
    rotation = load_int('cam_rotation')
    AF = load_bool('cam_AFmode')
    camera = load_str('camera')


    if camera == 'imx519' and AF == True:
        autofocus = ' --autofocus '
    else:
        autofocus = ''

    if camera  == "usb_webcam":
        cmd = 'fswebcam -i 0 -r "1280x720" -F 5 --no-banner --jpeg 95 --save ' + filepath2
    else:
        cmd = 'libcamera-still -n --denoise off --sharpness 0 -o ' + filepath2 + ' -t ' + timeout  +' --shutter ' + shutter + ' --saturation ' + saturation + ' --contrast ' + contrast + ' --awbgains '+awbg_red + "," + awbg_blue + ' --gain ' + gain + ' -q ' + str(quality) + autofocus + ' >/dev/null 2>&1'
    #    cmd = 'libcamera-still -n --denoise off --sharpness 0 -o ' + filepath2 + ' -t ' + timeout  +' --shutter ' + shutter + ' --saturation ' + saturation + ' --contrast ' + contrast + ' --awbgains '+awbg_red + "," + awbg_blue + ' --gain ' + gain + ' -q ' + str(quality) + autofocus
        
    system(cmd)
    return cmd

def get_points(samples=1):
    from math import pi, sqrt, acos, atan2, cos, sin

    points = []
    phi = pi * (3. - sqrt(5.))
    for i in range(int(samples)):
        y = 1 - (i / float(samples - 1)) * 2
        radius = sqrt(1 - y * y)
        theta = phi * i
        x = cos(theta) * radius
        z = sin(theta) * radius
        r=sqrt(x*x+y*y+z*z)
        theta_neu=acos(z/r)*180/pi
        phi_neu=atan2(y,x)*180/pi
        points.append((theta_neu-90,phi_neu))
    points.sort()
    return points

def create_coordinates(angle_min, angle_max,point_count):
    point_count_final=point_count
    if angle_max < angle_min:
        a = angle_min
        angle_min = angle_max
        angle_max = a
    point_count=point_count*90/(angle_max-angle_min)
    actual_points=0
    while actual_points<point_count_final:
        points=get_points(point_count)
        filtered=[]
        for x,y in points:
            if x>angle_min and x<angle_max and len(filtered)<point_count_final:
                filtered.append((x,y))
        actual_points=len(filtered)

        if point_count-actual_points>20:
            point_count=point_count+3
        else:
            point_count=point_count+1
    return filtered


def haversine_distance_deg(theta1, phi1, theta2, phi2):
    import numpy as np
    R = 1
    dtheta = np.radians(theta2 - theta1)
    dphi = np.radians(phi2 - phi1)

    theta1, phi1 = np.radians(theta1), np.radians(phi1)
    theta2, phi2 = np.radians(theta2), np.radians(phi2)

    a = np.sin(dtheta / 2) ** 2 + np.cos(theta1) * np.cos(theta2) * np.sin(dphi / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def sort_spherical_coordinates_deg(points_spherical_deg):
    import numpy as np
    from tsp_solver.greedy import solve_tsp

    points_spherical_deg = np.array(points_spherical_deg)  # Convert list of tuples to NumPy array

    n = len(points_spherical_deg)
    dist_matrix = np.zeros((n, n))

    # Calculate haversine distance for each pair of points
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance_deg(points_spherical_deg[i, 0], points_spherical_deg[i, 1],
                                          points_spherical_deg[j, 0], points_spherical_deg[j, 1])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Solve the TSP problem using the tsp_solver.greedy algorithm
    path = solve_tsp(dist_matrix)

    sorted_points_spherical_deg = points_spherical_deg[path]

    # Convert the sorted NumPy array back to a list of tuples
    return [tuple(point) for point in sorted_points_spherical_deg]
