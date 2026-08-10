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
        self.setup_layout("Real-world Application: Nature's Engineering", [
            "Nature often optimizes travel time efficiently.",
            "Peregrine falcons follow near-cycloid dive paths.",
            "Evolution favors geometry that minimizes strike time."
        ])
        
        # Create cycloid path
        # x = r(theta - sin(theta)), y = -r(1 - cos(theta))
        r = 0.6
        def cycloid_func(t):
            return np.array([r * (t - np.sin(t)), -r * (1 - np.cos(t)), 0])
        
        cycloid = ParametricFunction(cycloid_func, t_range=[0, 2*np.pi], color=BLUE)
        falcon = Dot(color=YELLOW)
        falcon_group = VGroup(cycloid, falcon)
        
        # Initial display: place cycloid group in area
        self.place_in_area(falcon_group, "B3", "E5", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        # Issue 38: Move dot animation along path
        self.play(MoveAlongPath(falcon, cycloid), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Issue 39: Text layout
        strike_time_text = Text("Minimizes Strike Time", font_size=20, color=GREEN)
        self.place_in_area(strike_time_text, 'D4', 'E6', scale_factor=0.7)
        
        self.play(FadeIn(strike_time_text))
        self.wait(2)
