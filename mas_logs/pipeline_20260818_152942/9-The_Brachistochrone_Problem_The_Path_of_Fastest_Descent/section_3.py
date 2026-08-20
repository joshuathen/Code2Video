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
        lecture_lines = [
            "The solution is a curve called a cycloid.",
            "It is traced by a rolling wheel's rim.",
            "This shape balances distance and speed perfectly."
        ]
        self.setup_layout("Defining the Shape: The Cycloid", lecture_lines)
        
        # Setup Animation Elements
        r = 0.5
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg")
        point = Dot(color="#00FFFF")
        
        # Cycloid Path
        cycloid = ParametricFunction(
            lambda t: np.array([r * (t - np.sin(t)), -r * (1 - np.cos(t)), 0]),
            t_range=[0, 2*PI], color="#FFD700"
        )
        
        # Grouped for layout constraint
        animation_group = VGroup(wheel, point, cycloid)
        self.place_in_area(cycloid, 'B2', 'C5', scale_factor=0.7)
        self.place_in_area(animation_group, 'B2', 'E5', scale_factor=0.65)
        
        # Parabola
        parabola = FunctionGraph(lambda x: -0.2 * x**2, x_range=[-1, 1], color="#FF4500")
        self.place_in_area(parabola, 'D2', 'E5', scale_factor=0.7)
        
        self.add(wheel, point)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(wheel), Write(point))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        
        # Simulate rolling
        tracker = ValueTracker(0)
        # Using the wheel asset
        wheel.add_updater(lambda m: m.move_to(self.grid['C2'] + np.array([tracker.get_value(), 0, 0])))
        point.add_updater(lambda m: m.move_to(wheel.get_center() + r * np.array([np.sin(tracker.get_value()/r), -np.cos(tracker.get_value()/r), 0])))
        
        self.play(Create(cycloid), tracker.animate.set_value(2*PI*r), run_time=3)
        
        # Cleanup
        wheel.clear_updaters()
        point.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        
        self.play(FadeIn(parabola), run_time=2)
        # Highlight cycloid smooth transition property using the wheel
        self.play(wheel.animate.set_color("#00FF00"), run_time=1)
        self.wait(2)
