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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("Application: The Laser-Guided Robot", [
            "Dexter the Robot follows the complex Folium of Descartes.",
            "He arrives at point three comma three on the curve.",
            "His laser must stay perfectly tangent to his path.",
            "Implicit differentiation calculates the laser's slope as negative one.",
            "As Dexter moves, the laser updates in real-time."
        ])

        # === Animation for Lecture Line 1 ===
        # Dexter the Robot follows the complex Folium of Descartes.
        coordinate_system = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Issue 44: Anchor the coordinate system to the grid area
        self.place_in_area(coordinate_system, 'B2', 'F6', scale_factor=0.9)
        
        # Define the Folium of Descartes loop (x^3 + y^3 = 6xy) parametrically
        # x(t) = 6t/(1+t^3), y(t) = 6t^2/(1+t^3)
        curve = ParametricFunction(
            lambda t: coordinate_system.c2p(6*t/(1+t**3), 6*t**2/(1+t**3)),
            t_range=[0.1, 10], color="#00CED1"
        )
        
        self.play(
            Create(coordinate_system),
            Create(curve),
            self.lecture[0].animate.set_color("#00CED1")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # He arrives at point three comma three on the curve.
        robot_point = Dot(color=WHITE)
        # Issue 45: Position robot_point on grid C4
        self.place_at_grid(robot_point, 'C4', scale_factor=0.5)
        
        # Visually snap the robot to (3,3) on the curve for mathematical consistency
        self.play(
            robot_point.animate.move_to(coordinate_system.c2p(3, 3)),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # His laser must stay perfectly tangent to his path.
        # Tangent at (3,3) has slope -1: y - 3 = -1(x - 3) => y = -x + 6
        laser_line = Line(
            coordinate_system.c2p(2.5, 3.5), 
            coordinate_system.c2p(3.5, 2.5), 
            color="#FF0000",
            stroke_width=4
        )
        dy_dx_label = Text("dy/dx", font_size=20, color="#FFFF00")
        self.place_at_grid(dy_dx_label, 'C5', scale_factor=1.0)
        
        self.play(
            Create(laser_line),
            FadeIn(dy_dx_label),
            self.lecture[2].animate.set_color("#FF0000")
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Implicit differentiation calculates the laser's slope as negative one.
        eq1 = Text("x³ + y³ = 6xy", font_size=22)
        eq2 = Text("dy/dx = (2y - x²)/(y² - 2x)", font_size=20)
        eq3 = Text("Slope at (3,3) = -1", font_size=22, color="#FFFF00")
        derivations_group = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.2)
        
        # Issue 43: Place derivations at A1 to avoid obstructing lecture
        self.place_at_grid(derivations_group, 'A1', scale_factor=0.6)
        
        self.play(
            FadeIn(derivations_group),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # As Dexter moves, the laser updates in real-time.
        t_tracker = ValueTracker(1.0)
        
        def get_pos(t):
            return np.array([6*t/(1+t**3), 6*t**2/(1+t**3), 0])
            
        def get_slope(t):
            # Parametric derivative: (dy/dt) / (dx/dt) = (2t - t^4) / (1 - 2t^3)
            return (2*t - t**4) / (1 - 2*t**3 + 1e-9)

        # Updaters for real-time tracking
        robot_point.add_updater(lambda d: d.move_to(
            coordinate_system.c2p(*get_pos(t_tracker.get_value())[:2])
        ))
        
        def update_laser(l):
            t = t_tracker.get_value()
            pos = get_pos(t)
            m = get_slope(t)
            # Create a unit direction vector for the tangent line
            length = 1.2
            dir_vec = np.array([1, m, 0])
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            l.put_start_and_end_on(
                coordinate_system.c2p(*(pos[:2] - dir_vec[:2] * length/2)),
                coordinate_system.c2p(*(pos[:2] + dir_vec[:2] * length/2))
            )
            
        laser_line.add_updater(update_laser)
        dy_dx_label.add_updater(lambda m: m.next_to(robot_point, RIGHT, buff=0.1))

        # Perform the movement along the loop
        self.play(
            t_tracker.animate.set_value(1.4),
            self.lecture[4].animate.set_color(WHITE),
            run_time=2,
            rate_func=linear
        )
        self.play(
            t_tracker.animate.set_value(0.6),
            run_time=3,
            rate_func=linear
        )
        self.wait(2)
