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
        lecture_lines = ["To solve for x, substitute column.", "This new parallelogram represents the numerator.", "Its area scales with x."]
        self.setup_layout("The Geometry of the Numerator", lecture_lines)
        
        # Define base vectors for the parallelogram
        a1 = np.array([1, 2, 0])
        a2 = np.array([2, 1, 0])
        b = np.array([1.5, 2.5, 0])
        
        # Setup axes
        axes = Axes(x_range=[-1, 4], y_range=[-1, 4], axis_config={"include_tip": False})
        self.place_at_grid(axes, "D3", scale_factor=0.4)
        
        # Parallelogram Mobjects
        orig_para = Polygon(ORIGIN, a1, a1+a2, a2, color=BLUE, fill_opacity=0.3)
        
        # Asset for new parallelogram
        new_para = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        self.place_in_area(new_para, 'C4', 'E6', scale_factor=0.5)
        
        area_formula = MathTex(r"Area \propto x")
        self.place_at_grid(area_formula, 'B5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(orig_para)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(new_para))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.add(area_formula)
        self.play(new_para.animate.scale(1.2))
        self.wait(2)
