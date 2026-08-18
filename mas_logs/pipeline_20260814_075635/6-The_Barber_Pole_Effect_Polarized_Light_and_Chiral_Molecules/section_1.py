from manim import *
import os

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
        self.setup_layout("Prerequisite: Polarization and Malus's Law", [
            "Light vibrates in a single plane.",
            "Filter blocks other orientations.",
            "Only vertical wave passes through."
        ])
        
        # Colors
        color_1 = "#00BFFF" # DeepSkyBlue
        color_2 = "#FF4500" # OrangeRed
        color_3 = "#FFFF00" # Yellow
        
        # Objects
        # [Asset: Polarization_Vector]
        vec = Arrow(start=ORIGIN, end=UP*1.5, color=color_1, buff=0)
        self.place_at_grid(vec, 'B4', scale_factor=0.6)
        
        # [Asset: Wave_Passing]
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3*x), x_range=[-2, 2], color=color_2)
        self.place_in_area(wave, 'D3', 'E4', scale_factor=0.6)
        
        # Malus's Law
        equation = MathTex(r"I = I_0 \cos^2(\theta)", color=color_3)
        self.place_at_grid(equation, 'F2', scale_factor=1.0)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg]
        filter_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        self.place_at_grid(filter_asset, 'A3', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_1)
        self.play(Create(vec), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_2)
        polarizer = Square(side_length=2, color=WHITE).set_fill(WHITE, opacity=0.1)
        self.place_at_grid(polarizer, 'B3', scale_factor=0.5)
        self.play(FadeIn(filter_asset), Rotate(polarizer, angle=PI/4), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_3)
        self.play(Write(equation), run_time=1)
        self.play(Indicate(equation), run_time=1)
        self.wait(1)
