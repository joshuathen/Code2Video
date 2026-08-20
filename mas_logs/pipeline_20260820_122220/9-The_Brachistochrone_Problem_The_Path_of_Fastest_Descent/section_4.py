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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Solution: The Cycloid", [
            "The solution is a curve called a cycloid.",
            "A rolling circle traces this unique path.",
            "It perfectly balances distance and high velocity."
        ])
        
        # --- Cycloid setup ---
        # Adjust layout based on feedback: C2 to F6, scale 0.95
        def cycloid_path(t):
            return np.array([t - np.sin(t), -np.cos(t), 0])
        
        # Draw cycloid
        cycloid = ParametricFunction(cycloid_path, t_range=[0, 2*PI], color=WHITE)
        self.place_in_area(cycloid, 'C2', 'F6', scale_factor=0.95)
        
        # Rolling circle using Asset
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=BLUE)
        tracker = ValueTracker(0)
        
        # Tethering the circle updater to the cycloid's path logic
        circle.add_updater(lambda c: c.move_to(cycloid.point_from_proportion(tracker.get_value() / (2 * PI))))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(cycloid))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.add(circle)
        self.play(tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.wait(1)
