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
        lines = ["Define interface and coordinate system.", "Specify points A and B.", "Express total time as function.", "Locate the global time minimum.", "This determines the light's path."]
        self.setup_layout("Mathematical Modeling", lines)
        
        # Pre-load Assets
        interface_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/interface.svg")
        coord_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coordinate.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        points_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/points.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(interface_icon, "B2", scale_factor=0.5)
        self.play(FadeIn(interface_icon))
        origin_label = Text("(0,0)", font_size=16, color=WHITE).next_to(interface_icon, DOWN, buff=0.1)
        self.play(Write(origin_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        self.place_at_grid(coord_icon, "B5", scale_factor=0.5)
        self.play(FadeIn(coord_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        formula = MathTex(r"T(y) = \frac{\sqrt{h_1^2 + y^2}}{v_1} + \frac{\sqrt{h_2^2 + (x-y)^2}}{v_2}", font_size=20)
        self.place_in_area(formula, "D2", "D5", scale_factor=1.0)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        self.place_at_grid(graph_icon, "E2", scale_factor=0.4)
        self.play(FadeIn(graph_icon))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ORANGE)
        self.place_at_grid(points_icon, "E5", scale_factor=0.4)
        self.play(FadeIn(points_icon))
        self.wait(2)
