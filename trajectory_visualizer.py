import itertools
import time

import pyglet
from pyglet.gl import *
from math import cos, sin, pi

from trajectory_generator import *


def main():
    seed = np.random.randint(10000)
    # seed = 5118
    np.random.seed(seed)
    print(seed)
    window = TrajectoryWindow()
    glClearColor(.5, .5, .5, 1.0)
    pyglet.clock.schedule_interval(window.my_tick, window.sim_dt)
    pyglet.app.run()


class TrajectoryWindow(pyglet.window.Window):
    def __init__(self):
        super(TrajectoryWindow, self).__init__(800, 600)
        # Order of derivative on r
        k_r = 4
        # Order of polynomials
        n = 5

        ndims = 2

        # Keyframes stored as (x,y,z,psi,t) triples
        self.keyframes = [(-4.0, -4.0, 0.0),
                          (-4.0, 4.0, 0.2),
                          (4.0, 4.0, 0.4),
                          (4.0, -4.0, 0.6),
                          (-4.0, -4.0, 0.8)]

        self.keyframes = []
        t = 0.0
        rand_signs = {0:(-1, -1), 1:(-1, 1), 2:(1,1), 3: (1, -1)}

        for i in range(4):
            p = list(np.random.rand(2)*5) + np.array([1.0, 1.0])
            sx, sy = rand_signs[i]
            self.keyframes.append([sx*p[0], sy*p[1], t])
            t += 0.2
        end_node = copy.deepcopy(self.keyframes[0])
        end_node[-1] = t
        self.keyframes.append(end_node)

        tgen = TrajectoryGenerator(self.keyframes, n, ndims, k_r)

        x = tgen.generate_trajectory()

        self.polys = tgen.parse_polys_from_vector(x)
        for poly in self.polys:
            print(poly)

        glTranslatef(self.width / 2.0, self.height / 2.0, 0.0)
        scale = 40.0
        glScalef(scale, scale, 1.0)

        self.current_time = 0.0
        self.time_scale = 0.125
        self.max_time = self.keyframes[-1][-1]

        self.sim_dt = 1.0 / 60.0

    def my_tick(self, dt):
        self.clear()

        y_index_offset = len(self.polys) / 2
        index_str = ""

        # Draw keyframes
        for i in range(len(self.keyframes)):
            x, y = self.keyframes[i][:2]
            glPushMatrix()
            glTranslatef(x, y, 0.0)
            draw_circle(0.1)
            glPopMatrix()

            if self.current_time >= self.keyframes[i][-1]:
                index_str = "i: " + str(i) + " t: " + str(self.current_time)
                x_poly = self.polys[i]
                y_poly = self.polys[i + y_index_offset]

        x = eval_poly(x_poly, self.current_time)
        y = eval_poly(y_poly, self.current_time)

        glPushMatrix()
        glTranslatef(x, y, 0.0)
        draw_circle(0.2, [0, 255, 0])
        glPopMatrix()

        glPushMatrix()
        x = str(round(x, 4)).zfill(4)
        y = str(round(y, 4)).zfill(4)
        x_val = pyglet.text.Label(text="x: {}".format(x), x=135, y=225)
        y_val = pyglet.text.Label(text="y: {}".format(y), x=225, y=225)
        glScalef(.03, .03, 1.0)
        x_val.draw()
        y_val.draw()
        glPopMatrix()
        self.current_time += dt * self.time_scale
        self.current_time %= self.max_time


def eval_poly(poly, t):
    return sum([poly[i] * t ** i for i in range(len(poly))])


def draw_circle(radius, color=(255, 0, 0)):
    number_of_triangles = 12
    angle = (2 * pi) / number_of_triangles

    x = [radius * cos(angle * i) for i in range(number_of_triangles + 1)]
    y = [radius * sin(angle * i) for i in range(number_of_triangles + 1)]
    points = [0, 0]
    points += list(itertools.chain(*zip(x, y)))

    pyglet.graphics.draw(number_of_triangles + 2, GL_TRIANGLE_FAN,
                         ('v2f', points),
                         ('c3B', color * (number_of_triangles + 2)))


if __name__ == "__main__":
    main()
