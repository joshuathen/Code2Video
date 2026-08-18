from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Derivative Function", 
                          ["The derivative acts as a transformation machine.", 
                           "It maps x to the curve's slope.", 
                           "This creates a new velocity function.", 
                           "Think of a robot tracking speed.", 
                           "Velocity is plotted over time."])
        
        # Setup Axes and Curves
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 3, 1], x_length=4, y_length=3)
        f = lambda x: 0.5 * x**2 + 0.5
        f_prime = lambda x: x
        
        curve = axes.plot(f, color=BLUE)
        deriv_curve = axes.plot(f_prime, color="#9B59B6")
        graph_group = VGroup(curve, deriv_curve)
        
        # Load asset: robot
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Fix 37 / Critic requirements
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        self.place_in_area(graph_group, 'B3', 'E4', scale_factor=0.6)
        
        self.add(axes, graph_group)

        # Logic for animation
        robot_tracker = Dot(color=YELLOW) # Proxy for robot
        self.place_at_grid(robot, 'C4', scale_factor=0.5)
        self.add(robot)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F1C40F")
        # Moving along curve
        path_points = [axes.c2p(x, f(x)) for x in np.linspace(-1.5, 1.5, 50)]
        path = VMobject()
        path.set_points_smoothly(path_points)
        
        self.play(MoveAlongPath(robot, path), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#F1C40F")
        # Show mapping
        indicator = Line(start=axes.c2p(0, 0), end=axes.c2p(0, 0), color=WHITE)
        self.add(indicator)
        
        def update_indicator(mob):
            # robot is at axes.c2p(x, f(x))
            # Find current x on axes
            x = axes.p2c(robot.get_center())[0]
            mob.put_start_and_end_on(axes.c2p(x, f(x)), axes.c2p(x, f_prime(x)))
        
        indicator.add_updater(update_indicator)
        self.wait(2)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F1C40F")
        f_prime_label = Text("f'(x)", font_size=24, color="#9B59B6")
        self.place_at_grid(f_prime_label, 'E5', scale_factor=0.7)
        self.wait(2)
