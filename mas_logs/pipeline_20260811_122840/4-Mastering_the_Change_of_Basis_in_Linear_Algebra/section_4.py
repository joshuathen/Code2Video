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
        self.setup_layout("Summary & Geometric Intuition", [
            "Change of basis is space relabelling.",
            "The vector itself stays the same.",
            "P-inverse converts back to original basis."
        ])
        
        # Load Assets
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        translator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/translator.svg")
        
        # Create standard and transformed axes
        std_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_ticks": False}).scale(0.4)
        transformed_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_ticks": False, "color": BLUE}).scale(0.4)
        transformed_axes.rotate(PI/6)
        
        # Vector
        vec = Arrow(start=ORIGIN, end=np.array([1, 1, 0]), color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_in_area(globe, 'B4', 'C6', scale_factor=0.3)
        self.play(FadeIn(globe), Create(std_axes), Create(transformed_axes))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(YELLOW)
        # Using requested area from issue 23, 24, 25 for consistency and grid usage
        self.place_in_area(vec, 'B4', 'E6', scale_factor=0.7) 
        self.play(GrowArrow(vec))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(GREEN)
        self.play(transformed_axes.animate.rotate(-PI/6), run_time=2)
        self.place_in_area(translator, 'D4', 'E6', scale_factor=0.3)
        self.play(FadeIn(translator))
