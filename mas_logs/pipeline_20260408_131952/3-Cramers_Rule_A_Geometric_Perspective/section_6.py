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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup with requested lecture lines
        lecture_lines = [
            "Cramer's Rule uses these area ratios for any variable.", 
            'Solving for y, we find a result of three.', 
            'Vic follows the paths and reaches his target!'
        ]
        self.setup_layout("The Formula and Conclusion", lecture_lines)

        # Colors
        COLOR_Y = "#FF0000"  # Red
        COLOR_X = WHITE
        VIC_COLOR = PINK

        # Vector data
        v1_coords = np.array([1, 0.5, 0])
        v2_coords = np.array([0.5, 1.5, 0])
        x_val, y_val = 2, 3
        b_coords = x_val * v1_coords + y_val * v2_coords

        # === Animation for Lecture Line 1 ===
        # x = det(b, v2) / det(v1, v2), y = det(v1, b) / det(v1, v2)
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        x_formula = Text("x = det(b, v2) / det(v1, v2)", font_size=24, color=WHITE)
        self.place_in_area(x_formula, 'A2', 'A5', scale_factor=0.9)
        
        y_formula = Text("y = det(v1, b) / det(v1, v2)", font_size=24, color=WHITE)
        self.place_in_area(y_formula, 'B2', 'B5', scale_factor=0.9)
        
        self.play(Write(x_formula), Write(y_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display numerical results: x = 10/5=2 (white) and y = 15/5=3 (red)
        self.play(self.lecture[1].animate.set_color(COLOR_Y))
        
        x_calc = Text("x = 10 / 5 = 2", font_size=24, color=WHITE)
        self.place_in_area(x_calc, 'C2', 'C5', scale_factor=0.9) # Fix Issue 45
        
        y_calc = Text("y = 15 / 5 = 3", font_size=24, color=COLOR_Y)
        self.place_in_area(y_calc, 'D2', 'D5', scale_factor=0.9) # Fix Issue 43
        
        self.play(Write(x_calc), Write(y_calc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualizing Vic [Asset] moving along path to Target [Asset]
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        plane = NumberPlane(
            x_range=[0, 6, 1], 
            y_range=[0, 7, 1],
            background_line_style={"stroke_opacity": 0.2}
        )
        
        # Vectors and Labels
        v1_arrow = Arrow(plane.c2p(0,0,0), plane.c2p(*v1_coords), buff=0, color=BLUE)
        v2_arrow = Arrow(plane.c2p(0,0,0), plane.c2p(*v2_coords), buff=0, color=COLOR_Y)
        v1_label = Text("v1", font_size=16, color=BLUE).next_to(v1_arrow.get_end(), DOWN, buff=0.1)
        v2_label = Text("v2", font_size=16, color=COLOR_Y).next_to(v2_arrow.get_end(), LEFT, buff=0.1)
        
        # Assets: Vic and Target
        vic = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/vic.svg")
        vic.set_color(VIC_COLOR).scale(0.2)
        vic.move_to(plane.c2p(0,0,0))
        
        target = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/target.svg")
        target.set_color(YELLOW).scale(0.2)
        target.move_to(plane.c2p(*b_coords))
        
        b_label = Text("b", font_size=16, color=YELLOW).next_to(target, UR, buff=0.05)
        
        visual_group = VGroup(plane, v1_arrow, v2_arrow, v1_label, v2_label, vic, target, b_label)
        self.place_in_area(visual_group, 'E2', 'F6', scale_factor=0.5) # Fix Issue 44
        
        self.play(Create(plane))
        self.play(Create(v1_arrow), Create(v2_arrow), Write(v1_label), Write(v2_label))
        self.play(FadeIn(target), Write(b_label))
        self.play(FadeIn(vic))
        
        # Path movement
        # 1. 2 * V1
        path1_end = 2 * v1_coords
        self.play(vic.animate.move_to(plane.c2p(*path1_end)), run_time=1.5, rate_func=linear)
        
        # 2. 3 * V2
        self.play(vic.animate.move_to(plane.c2p(*b_coords)), run_time=2, rate_func=linear)
        
        self.wait(3)
