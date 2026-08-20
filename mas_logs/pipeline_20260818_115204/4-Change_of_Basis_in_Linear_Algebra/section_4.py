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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Step 1: Identify your basis vectors.",
            "Step 2: Build the transition matrix.",
            "Step 3: Apply matrix to convert coordinates."
        ]
        self.setup_layout("Step-by-Step Computational Workflow", lecture_lines)
        
        # Mobjects
        P_inv = MathTex("P^{-1}").set_color(WHITE)
        X = MathTex("X").set_color(WHITE)
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg").set_color(WHITE)
        
        # Combined group for Step 1
        step1_group = VGroup(P_inv, X, calc).arrange(RIGHT, buff=0.3)
        self.place_in_area(step1_group, 'B2', 'B5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Step 1: Identify your basis vectors.
        self.play(FadeIn(step1_group))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Build the transition matrix.
        prod = MathTex("P^{-1}", "X").set_color(WHITE)
        self.place_at_grid(prod, 'D2', scale_factor=1.0)
        
        self.play(ReplacementTransform(step1_group, prod))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Apply matrix to convert coordinates.
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0.1)
        X_prime = MathTex("X'").set_color("#00FF00")
        comp = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg").set_color("#00FF00")
        
        final_group = VGroup(prod, arrow, X_prime, comp).arrange(RIGHT, buff=0.2)
        self.place_at_grid(arrow, 'D3', scale_factor=0.8)
        self.place_at_grid(X_prime, 'D4', scale_factor=1.0)
        self.place_at_grid(comp, 'D5', scale_factor=1.0)
        
        self.play(FadeIn(arrow), FadeIn(X_prime), FadeIn(comp))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
