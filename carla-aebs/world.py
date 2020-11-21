import carla
import math
import numpy as np
import os
import pygame
import queue
import torch
import torchvision.transforms as T

from PIL import Image

from utils.pid import PID
from models.icad.vae import VAE
from models.perception.model import PerceptionNet


def image_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


class World:
    # This class setups the world corresponding to AEBS testing with two vehicle actors
    def __init__(
        self,
        host='localhost',
        port=2000,
        map_type='Town01',
        gui=True, collect=True,
        collect_path=None,
        resX=800,
        resY=600,
        testing=False,
        perception_chkpt=None,
        vae_chkpt=None,
        calibration_scores=None
    ):
        self.collect = collect
        self.collect_path = collect_path
        if self.collect and self.collect_path is None:
            raise Exception('The `collect_path` param must be specified when collect=True')
        self.host = host
        self.port = port
        self.map_type = map_type
        self.resX = resX
        self.resY = resY
        self.gui = gui
        self.testing = testing
        self.episode = 1
        self.sensor_list = []

        # Book-keeping variables
        self.computed_distances = []
        self.gt_distances = []
        self.p_values = []

        # Create the client
        try:
            # TODO: Update the code to automatically start the carla server in the background
            # if not already running, thus bypassing the need for this check
            self.client = carla.Client(host, port)
            self.client.set_timeout(4.0)
        except Exception:
            raise Exception('Make sure the Carla server is running!')

        maps = self.client.get_available_maps()
        map_ = f'/Game/Carla/Maps/{self.map_type}'
        if map_ not in maps:
            raise Exception(f'The requested map {self.map_type} is not available!')

        # Get the world instance
        self.world = self.client.load_world(self.map_type)
        self.bp_library = self.world.get_blueprint_library()

        # Actor variables
        self.leading_car = None
        self.lagging_car = None

        if self.testing:
            # Perception LEC
            if perception_chkpt is None:
                raise ValueError('A valid `perception_chkpt` must be set when testing')
            self.perception_lec = PerceptionNet()
            self.perception_lec.load_state_dict(torch.load(perception_chkpt)['model'])
            self.perception_lec.eval()
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])

            if vae_chkpt is None:
                raise ValueError('A valid `vae_chkpt` must be set when testing')
            self.vae = VAE()
            self.vae.load_state_dict(torch.load(vae_chkpt)['model'])
            self.vae.eval()

            # Load the calibration scores
            self.calibration_scores = np.load(calibration_scores)

        # Set the pygame display
        self.display = None
        if self.gui:
            pygame.init()
            pygame.font.init()
            self.surface = None
            self.display = pygame.display.set_mode((self.resX, self.resY), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Carla AEBS")

    def render(self, image):
        array = image_to_array(image)
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if surface is not None:
            self.display.blit(surface, (0, 0))

    def init(self, initial_velocity, initial_distance, precipitation=0):
        # Create a new queue for each initialization
        self.image_queue = queue.Queue()
        self.collision_queue = queue.Queue()
        self.computed_distances = []
        self.gt_distances = []
        self.p_values = []

        # Setup world lighting conditions (like precipitation etc.)
        # TODO: Make these params as kwargs
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=precipitation,
            sun_altitude_angle=70.0,
            precipitation_deposits=precipitation
        )
        self.world.set_weather(weather)

        # Setup vehicles and sensors
        self._setup_vehicles()
        self._setup_sensors()

        if self.lagging_car is None:
            # This might happen due to failure in spawning
            return None, None, "FAILED"

        # Adjust some settings based on the experimental setup in the paper
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # Setup PID control to track the speed of the car to the initial speed
        self.pid_controller = PID(Kp=3.0, Ki=0.001, Kd=0)
        while True:
            velocity = self.lagging_car.get_velocity().y
            throttle = self.pid_controller.step(initial_velocity, velocity)
            self.lagging_car.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=0,
                    brake=0
                )
            )
            self.world.tick()
            img = self.image_queue.get()
            if self.gui:
                self.clock.tick()
                self.render(img)
                pygame.display.flip()
            if self.get_groundtruth_distance() <= initial_distance:
                break

        self.image_data = []
        self.distance_data = []
        print(f'Reset complete for episode: {self.episode}. Reset Distance: {self.get_groundtruth_distance()}, Reset Velocity: {velocity}')
        return self.get_groundtruth_distance(), velocity, "SUCCESS"

    def step(self, brake=0, throttle=0, steer=0):
        # Resumes simulation execution after the world is set up
        # to the desired parameters
        self.lagging_car.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
        )
        self.world.tick()
        img = self.image_queue.get()
        if self.gui:
            self.clock.tick()
            self.render(img)
            pygame.display.flip()
        if self.collect:
            self.image_data.append(img)
            self.distance_data.append(self.get_groundtruth_distance())

        velocity = self.lagging_car.get_velocity().y
        gt_dist = self.get_groundtruth_distance()
        dist = gt_dist
        if self.testing:
            pil_image = Image.fromarray(image_to_array(img))
            frame = self.transform(pil_image).unsqueeze(0)
            dist = self.get_distance(frame)
            p_value = self.get_pvalue(frame)
            self.p_values.append(p_value)
        self.computed_distances.append(dist)
        self.gt_distances.append(gt_dist)

        # Check if the episode needs to end
        has_stopped = velocity <= 0
        has_collided = not self.collision_queue.empty()
        stop_episode = has_stopped or has_collided
        if not stop_episode:
            return dist, velocity, 0, "RESUME"

        # In case of a collision or vehicle stoppage, compute the reward
        if has_collided:
            collision = self.collision_queue.get().normal_impulse
            reward = -200.0 - math.sqrt(collision.x**2 + collision.y**2 + collision.z**2) / 100.0
            print("Collision: {}".format(reward))
        elif has_stopped:
            too_far_reward = -((dist - 3.0) / 250.0 * 400 + 20) * (dist > 3.0) 
            too_close_reward = - 20.0 * (dist < 1.0)
            reward = too_far_reward + too_close_reward
            print(f"Stop: {reward}, Distance: {dist}")

        # If collection is enabled, save the image data to disk
        if self.collect:
            # Setup the directory for storing images
            write_path = os.path.join(self.collect_path, f'episode_{self.episode}')
            if not os.path.isdir(write_path):
                os.makedirs(write_path, exist_ok=True)
            # Save the generated data (images + gt)
            for i in self.image_data:
                array = image_to_array(i)
                img = Image.fromarray(array).resize((400, 300))
                img.save(os.path.join(write_path, f'episode_{self.episode}_{i.frame}.png'))        
            np.save(os.path.join(write_path, 'target'), self.distance_data)

        # Cleanup actors
        self.destroy_actors()
        print(f'Episode {self.episode} ended!')
        self.episode += 1
        return dist, velocity, reward, "DONE"

    def _setup_vehicles(self):
        # This method creates 2 vehicle actors
        # for setting up the collision detection world
        leading_car_bp = self.bp_library.find('vehicle.audi.a2')
        lead_transform = carla.Transform(carla.Location(x=393.5, y=320.0, z=0.0), carla.Rotation(yaw=90))
        self.leading_car = self.world.spawn_actor(leading_car_bp, lead_transform)

        lagging_car_bp = self.bp_library.find('vehicle.audi.tt')
        lag_transform = carla.Transform(carla.Location(x=391.5, y=10.0, z=0.02), carla.Rotation(yaw=89.6))
        self.lagging_car = self.world.try_spawn_actor(lagging_car_bp, lag_transform)

    def _setup_sensors(self):
        # This method attaches the camera sensor to the lagging car
        camera_bp = self.bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.resX))
        camera_bp.set_attribute('image_size_y', str(self.resY))
        camera_bp.set_attribute('fov', '100')
        camera = self.world.try_spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=0.8,y=0.0,z=1.7), carla.Rotation(yaw=0.0)),
            attach_to=self.lagging_car
        )
        self.start_collection = False

        collision_bp = self.bp_library.find('sensor.other.collision')
        collision = self.world.try_spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.lagging_car
        )

        # Setup the sensor callbacks
        camera.listen(lambda data: self.handle_camera(data))
        collision.listen(lambda data: self.handle_collision(data))
        self.sensor_list.extend([camera, collision])

    def handle_camera(self, data):
        self.image_queue.put(data)

    def handle_collision(self, event):
        self.collision_queue.put(event)

    def get_groundtruth_distance(self):
        # Compute the distance between the lagging and the leading cars
        # While computing this distance we need to consider bounding boxes as well
        distance = self.leading_car.get_location().y - self.lagging_car.get_location().y \
                - self.lagging_car.bounding_box.extent.x - self.leading_car.bounding_box.extent.x
        return distance

    def get_distance(self, frame):
        # Compute the distance from the perception LEC
        with torch.no_grad():
            # Scale the obtained distances
            dist = self.perception_lec(frame).squeeze().item() * 120
        return dist

    def get_pvalue(self, input):
        with torch.no_grad():
            non_conformity_score = self.vae.get_non_conformity_score(input).item()
            p_value = np.sum(non_conformity_score <= self.calibration_scores) / self.calibration_scores.shape[-1]
        return p_value

    def destroy_actors(self):
        # This method destroys all the actors
        for sensor in self.sensor_list:
            sensor.destroy()
        self.lagging_car.destroy()
        self.leading_car.destroy()
        self.sensor_list = []


if __name__ == "__main__":
    world = World(collect=True, collect_path='/home/lexent/carla_simulation', resX=800, resY=600, gui=True)
    initial_distance = np.random.normal(100, 1)
    initial_velocity = np.random.uniform(25, 28)
    world.reset(initial_velocity, initial_distance)
