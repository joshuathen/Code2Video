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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Scaling by 1/s grows pieces by N.", "Relationship follows: N equals s to power D.", "Fractal dimension D is log N over log s."]
        self.setup_layout("Prerequisite: The Power Law of Scaling", lecture_lines)
        
        # Define visual elements
        formula = MathTex("N", "=", "s^D", color=YELLOW)
        dimension_formula = MathTex("D", "=", "\\frac{\\log(N)}{\\log(s)}", color=YELLOW)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg]
        # Using SVG
        try:
            scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        except:
            scale_icon = Circle(radius=0.3, color=BLUE)

        # Grouping for area utilization
        group_equations = VGroup(formula, dimension_formula, scale_icon)
        
        # Placement
        self.place_at_grid(formula, 'B2', scale_factor=1.2)
        self.place_at_grid(dimension_formula, 'D2', scale_factor=1.0)
        self.place_at_grid(scale_icon, 'E5', scale_factor=0.6) # peripheral quadrant
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(scale_icon))
        
        # Growth visualization
        s_tracker = ValueTracker(1)
        self.play(s_tracker.animate.set_value(3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Write(dimension_formula))
        self.wait(2)
