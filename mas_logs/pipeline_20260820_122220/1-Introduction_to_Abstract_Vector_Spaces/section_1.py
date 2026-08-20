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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are more than just arrows in space.",
            "We strip away geometry for abstract rules.",
            "Focus on addition and scalar scaling.",
            "Think of recipes for mixing colors.",
            "Or complex tasks scheduled over time."
        ]
        self.setup_layout("From Concrete to Abstract: The Shift", lecture_lines)
        
        # Colors
        concrete_color = "#FFD700"
        abstract_color = "#00BFFF"
        highlight_color = "#FF00FF"

        # Assets
        paint_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paint.svg")
        clock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")

        # Create elements
        concrete_text = Text("Concrete World", color=WHITE, font_size=24)
        abstract_text = Text("Abstract World", color=WHITE, font_size=24)
        transition_arrow = Arrow(LEFT, RIGHT, color=highlight_color)

        # Positioning with required fixes
        self.place_at_grid(concrete_text, 'B2', scale_factor=0.55)
        self.place_at_grid(abstract_text, 'B5', scale_factor=0.55)
        
        # Place icons next to text
        paint_icon.next_to(concrete_text, UP, buff=0.2).scale(0.5)
        clock_icon.next_to(abstract_text, UP, buff=0.2).scale(0.5)
        
        self.add(concrete_text, abstract_text, paint_icon, clock_icon)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(concrete_color))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(abstract_color))
        self.place_in_area(transition_arrow, 'C2', 'C5', scale_factor=0.8)
        self.play(GrowArrow(transition_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(highlight_color))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(concrete_color))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(abstract_color))
        self.wait(1)
