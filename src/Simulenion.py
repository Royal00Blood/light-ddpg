from DistanceBW2points import DistanceBW2points
from deviation_angle import deviation_angle as d_ang
import random
import matplotlib.pyplot as plt
import math as m
import numpy as np
import threading
from threading import *


class Env:

    def __init__(self):
        self.reset_env()
        self.__areal = 0.1  # граница области назначения
        self.__dist = 11.0  # граница полигона

    def reset_env(self):
        self.__set_state()
        self.__set_move()
        self.__generate_point()
        self.__res_reward_settings()
        self.__status_env = False

        # Goal movement settings
        self.__time = 0.1
        self.__trajectory = random.choice([1, 2, 3])
        self.__trajFlag = 1
        self.__d_angl_rad = 0.0
        self.__angl = 0.0
        self.__u = self.__goal[0]
        self.__v = self.__goal[1]
        self.__count_goal_time = 0
        # Plots value
        self.__velocity, self.__ang_velocity = [], []
        self.__Xr_list, self.__Yr_list = [], []
        self.__Xg_list, self.__Yg_list = [], []
        self.__number = 0
        self.__old_point = 10
        self.__delta_angle = 0.0
        self.__delta_angle_old = 0.0
        self.__time_sum = 0.0

    def __generate_point(self):
        point = random.choice([1, 2, 3, 4])
        if point == 1:
            self.__goal = [2.0, 4.0]
        if point == 2:
            self.__goal = [4.0, 5.0]
        if point == 3:
            self.__goal = [5.0, 3.0]
        if point == 4:
            self.__goal = [6.0, 4.0]
        # print("Goal x: {} y: {}".format(self.__goal[0], self.__goal[1]))

    def __set_state(self, x=0.0, y=0.0, z=0.0, xq=0.0, yq=0.0, zq=0.0, wq=0.0):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__Quat_x = xq
        self.__Quat_y = yq
        self.__Quat_z = zq
        self.__Quat_w = wq

    def __set_move(self, vel_x=0.0, vel_y=0.0, vel_z=0.0, ang_vel_x=0.0, ang_vel_y=0.0, ang_vel_z=0.0):
        self.__x_lin = vel_x
        self.__y_lin = vel_y
        self.__z_lin = vel_z
        self.__x_ang = ang_vel_x
        self.__y_ang = ang_vel_y
        self.__z_ang = ang_vel_z

    def __res_reward_settings(self):
        self.__reward = 0
        self.__old_state = [0.0, 0.0]
        self.__old_goal = [self.__goal[0], self.__goal[1]]

    def __calc_state(self):
        self.__d_angl_rad += self.__z_ang * self.__time  # рад
        self.__x += self.__x_lin * m.cos(self.__d_angl_rad) * self.__time
        self.__y += self.__x_lin * m.sin(self.__d_angl_rad) * self.__time
        self.__z = 0.0

        self.__Quat_x = 0.0
        self.__Quat_y = 0.0
        self.__Quat_z = 1 * m.sin(self.__d_angl_rad / 2)
        self.__Quat_w = m.cos(self.__d_angl_rad / 2)
        thread = threading.Thread(self.__update_goal())
        thread.start()
        thread.join()
        #self.__update_goal()
        self.__update_plots()

    def __update_plots(self):
        self.__Xr_list.append(self.__x)
        self.__Yr_list.append(self.__y)
        self.__Xg_list.append(self.__goal[0])
        self.__Yg_list.append(self.__goal[1])

    def __save_velocity(self, vel, ang_vel):
        self.__velocity.append(vel)
        self.__ang_velocity.append(ang_vel)
        self.__time_sum += self.__time

    def set_new_move(self, action):
        self.__set_move(vel_x=action[0], ang_vel_z=action[1])
        self.__calc_state()
        self.__save_velocity(vel=action[0], ang_vel=action[1])
        self.__new_reward()
        self.__check_env()

    def __update_goal(self):
        if self.__trajectory == 1:
            if self.__trajFlag == 1:
                if 0 < self.__goal[0] < 10 or 0 < self.__goal[1] < 10:
                    self.__goal[0] += 0.1
                    self.__goal[1] += 0.1
                else:
                    self.__trajFlag = 0
            else:
                if self.__goal[0] > 2 or self.__goal[1] > 2:
                    self.__goal[0] -= 0.1
                    self.__goal[1] -= 0.1
                else:
                    self.__trajFlag = 1
        elif self.__trajectory == 2:
            self.__goal[0] = 3 * m.cos(self.__angl) + self.__u
            self.__goal[1] = 3 * m.sin(self.__angl) + self.__v
            self.__angl += 0.01
        elif self.__trajectory == 3:
            self.__goal[0] += random.triangular(-0.2, 0.2, 0.01)
            self.__goal[1] += random.triangular(-0.2, 0.2, 0.01)
        else:
            print("Trajectory isn't")

    def get_new_state(self):
        state_env = [self.__x, self.__y, self.__z,
                     self.__Quat_x, self.__Quat_y, self.__Quat_z, self.__Quat_w,
                     self.__x_lin, self.__y_lin, self.__z_lin,
                     self.__x_ang, self.__y_ang, self.__z_ang,
                     self.__goal[0], self.__goal[1]]
        return [state_env, self.__reward, self.__status_env]

    def __new_reward(self):
        diff_distance = 0.0
        diff_angle_rad = 0.0
        dist_new = DistanceBW2points(self.__goal[0], self.__goal[1], self.__x, self.__y)
        dist_old = DistanceBW2points(self.__old_goal[0], self.__old_goal[1], self.__old_state[0],
                                     self.__old_state[1])

        self.__delta_angle = d_ang(self.__x, self.__y,
                                   self.__goal[0], self.__goal[1],
                                   self.__d_angl_rad).get_angle_dev()
        diff_angle_rad = abs(self.__delta_angle - self.__delta_angle_old)
        diff_distance = abs(dist_new.getDistance() - dist_old.getDistance())

        reward_angle = 0.0
        reward_dist = 0.0
        reward_ruel = 0.0

        if self.__count_goal_time == 4:
            print('Coordinate is goal')
            self.__reward = 400
            self.__status_env = True
        # if self.__goal[0] + self.__areal >= self.__x >= self.__goal[0] - self.__areal:
        #     if self.__goal[1] + self.__areal >= self.__y >= self.__goal[1] - self.__areal:
        #         print('Coordinate is goal')
        #         self.__reward = 400
        #         self.__status_env = True
        # elif min(self.laserScan)< 0.5 and min(self.laserScan) > 0:
        #     print('collision')
        #     self.reward = -200
        #     self.__reload_env()

        # elif self.__x >= self.__dist or self.__y >= self.__dist or self.__x <= -self.__dist or self.__y <= -self.__dist:
        #     print('Polygon out off range')
        #     self.__reward = -300
        #     self.__status_env = True
        else:
            if diff_angle_rad == 0.0:
                reward_angle = 17
            elif -np.pi / 2 < diff_angle_rad < np.pi / 2:
                reward_angle = -20.4 * diff_angle_rad + 16
            else:
                reward_angle = -17

            if 0.3 <= dist_new.getDistance() <= self.__areal:
                self.__count_goal_time += 1
            else:
                if dist_new.getDistance() < dist_old.getDistance():
                    reward_dist = 17
                else:
                    if diff_distance <= 1:
                        reward_dist = -25 * diff_distance + 15
                    else:
                        reward_dist = -17
            #  - если pursuit_value=0 преследование осуществляется
            pursuit_value = self.calculate_ruel_pursuit()
            if pursuit_value == 0:
                reward_ruel = 20
            else:
                reward_ruel = 16 * pursuit_value

            self.__reward = reward_angle + reward_dist + reward_ruel

            self.__old_goal = [self.__goal[0], self.__goal[1]]
            self.__old_state = [self.__x, self.__y]

            self.__delta_angle_old = self.__delta_angle

    def print_info(self):
        print(f"Iteration: {self.__number} Episod: {int(self.__time_sum * 10)} Rewad: {self.__reward} ")

    def __check_env(self):
        if self.__count_goal_time == 4:
            print('Coordinate is goal')
            self.graf_move()
            self.__status_env = True
        # if self.__goal[0] + self.__areal >= self.__x >= self.__goal[0] - self.__areal:
        #     if self.__goal[1] + self.__areal >= self.__y >= self.__goal[1] - self.__areal:
        #         print('Coordinate is goal')
        #         self.graf_move()
        #         self.__status_env = True
        # elif min(self.laserScan)< 0.5 and min(self.laserScan) > 0:
        # print('collision')
        # self.__reload_env()
        # elif self.__x >= self.__dist or self.__y >= self.__dist or self.__x <= -self.__dist or self.__y <= -self.__dist:
        #     print('Polygon out off range')
        #     self.__status_env = True
        else:
            self.__status_env = False

    def count_plots(self, number):
        self.__number = number

    def graf_move(self):
        time = self.__create_time_list()
        plt.figure(self.__number)
        plt.figure(figsize=(21, 7))
        # plot x(y)
        plt.subplot(1, 3, 1)
        plt.title("The trajectory of the robot")
        plt.plot(self.__Xr_list, self.__Yr_list)
        plt.plot(self.__Xg_list, self.__Yg_list)
        plt.plot(0, 0, marker="o", color='green')
        plt.plot(self.__goal[0], self.__goal[1], color='red')
        plt.xlabel("X")
        plt.ylabel("Y")
        # plot 2 v(t)
        plt.subplot(1, 3, 2)
        plt.title("Speed change")
        plt.xlabel("time")
        plt.ylabel("velocity")
        plt.grid()
        plt.plot(time, self.__velocity)
        # plot 3 w(t)
        plt.subplot(1, 3, 3)
        plt.title("Change in angular velocity")
        plt.xlabel("time")
        plt.ylabel("angle_velocity")
        plt.grid()
        plt.plot(time, self.__ang_velocity)

        plt.savefig('images/plot' + str(self.__number) + '.png', format='png')

    def __create_time_list(self):
        time_list = []
        value = 0
        for i in range(len(self.__velocity)):
            time_list.append(value)
            value += self.__time
        return time_list

    def calculate_ruel_pursuit(self):
        det_x = self.__goal[0] - self.__x
        det_y = self.__goal[1] - self.__y
        v = self.__velocity[-1]
        return (v * m.cos(self.__d_angl_rad) / det_x) - (v * m.sin(self.__d_angl_rad) / det_y)


def main():
    pass


if __name__ == '__main__':
    main()
