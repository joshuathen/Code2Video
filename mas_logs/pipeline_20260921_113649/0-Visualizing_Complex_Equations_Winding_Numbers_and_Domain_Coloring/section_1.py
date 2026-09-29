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
            "Complex plane maps inputs z to outputs w.",
            "Grid squares warp into complex shapes.",
            "Functions transform space like a rubber sheet.",
            "Mapping shows how space bends and flows.",
            "Visualizing functions reveals their hidden nature."
        ]
        self.setup_layout("Prerequisites: The Complex Plane", lecture_lines)
        
        # Elements
        axes = ComplexPlane(x_range=[-3, 3, 1], y_range=[-2, 2, 1], axis_config={"include_numbers": False})
        
        # Load asset
        sheet_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")
        z_point = sheet_asset.copy()
        
        z_label = MathTex("z", color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        self.place_in_area(axes, "B3", "F6", scale_factor=0.5)
        self.lecture[1].set_color(GREEN)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(z_point, "C4", scale_factor=0.3)
        z_label.next_to(z_point, UP, buff=0.1)
        self.play(FadeIn(z_point), Write(z_label))
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        modulus_line = Line(axes.c2p(0, 0), z_point.get_center(), color=RED)
        self.play(Create(modulus_line))
        self.lecture[3].set_color(RED)
        
        # === Animation for Lecture Line 5 ===
        self.play(Rotate(z_point, angle=PI/2, about_point=axes.c2p(0, 0)), 
                  Rotate(z_label, angle=PI/2, about_point=axes.c2p(0, 0)))
        self.lecture[4].set_color(PURPLE)
        
        self.wait(2)
