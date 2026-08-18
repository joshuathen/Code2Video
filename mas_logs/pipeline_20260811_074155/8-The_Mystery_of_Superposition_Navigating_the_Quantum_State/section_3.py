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
        lecture_lines = [
            "We use \"ket\" notation to label quantum states.",
            "The state psi is a sum of two possibilities.",
            "Coefficients alpha and beta scale these base states.",
            "Schrödinger’s cat is both alive and dead simultaneously.",
            "Superposition mathematically blends these two outcomes together."
        ]
        self.setup_layout("Defining Superposition (The Ket Notation)", lecture_lines)
        
        # Colors
        GREEN = "#00FF00"
        BLUE = "#00FFFF"
        YELLOW = "#FFFF00"
        WHITE = "#FFFFFF"

        # Asset path
        cat_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        # === Animation for Lecture Line 1 ===
        # Display the symbol '|ψ>' in the center in bright green (#00FF00).
        self.lecture[0].set_color(GREEN)
        # Fix: scale_factor=1.5 per Issue 35/44
        psi_ket = MathTex(r"\vert\psi\rangle", color=GREEN)
        self.place_in_area(psi_ket, "C2", "D5", scale_factor=1.5)
        self.play(Write(psi_ket))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show the components '|0>' and '|1>' appearing on either side of a plus sign.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        plus_m = MathTex(r"+", color=WHITE)
        state_0 = MathTex(r"\vert 0 \rangle", color=BLUE)
        state_1 = MathTex(r"\vert 1 \rangle", color=BLUE)
        comp_group = VGroup(state_0, plus_m, state_1).arrange(RIGHT, buff=0.5)
        self.place_in_area(comp_group, "C2", "D5", scale_factor=1.5)
        
        self.play(FadeOut(psi_ket), FadeIn(comp_group))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Add coefficients 'α' and 'β' in front of the basis states.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        alpha = MathTex(r"\alpha", color=YELLOW)
        beta = MathTex(r"\beta", color=YELLOW)
        
        full_sum = VGroup(
            alpha, 
            MathTex(r"\vert 0 \rangle", color=BLUE), 
            MathTex(r"+", color=WHITE), 
            beta, 
            MathTex(r"\vert 1 \rangle", color=BLUE)
        ).arrange(RIGHT, buff=0.25)
        self.place_in_area(full_sum, "C2", "D5", scale_factor=1.5)
        
        self.play(ReplacementTransform(comp_group, full_sum))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Schrödinger’s cat icons.
        # Replace create_cat with '/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png'.
        # Move cats to area 'A3'-'B4' (scale 1.2) to avoid formula overlap.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        
        cat_alive = ImageMobject(cat_asset_path).set_opacity(0.5)
        cat_dead = ImageMobject(cat_asset_path).set_opacity(0.5).rotate(-90*DEGREES)
        
        cats = Group(cat_alive, cat_dead).arrange(RIGHT, buff=1.0)
        self.place_in_area(cats, "A3", "B4", scale_factor=1.2)
        
        self.play(
            FadeIn(cats, shift=UP*0.3)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Fade in the full equation.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        
        final_eq = MathTex(r"\vert\psi\rangle = \alpha\vert 0 \rangle + \beta\vert 1 \rangle", color=GREEN)
        self.place_in_area(final_eq, "F2", "F5", scale_factor=1.2)
        
        self.play(
            FadeOut(cats),
            FadeOut(full_sum),
            FadeIn(final_eq)
        )
        self.wait(3)
        self.lecture[4].set_color(WHITE)
