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
        # 1. Setup Layout
        title_text = "Prerequisite: The Harmonic Series and Convergence"
        lecture_lines = [
            "Consider the infinite sum of one over n.",
            "This harmonic series grows forever, never reaching a limit.",
            "But squaring the denominator makes the total sum converge."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Start Animations
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Display the harmonic series formula in white
        harmonic_formula = MathTex(r"\sum_{n=1}^{\infty} \frac{1}{n}", color=WHITE)
        # Resolved Issue 35: Using 'B4' and scale 1.0 for better positioning
        self.place_at_grid(harmonic_formula, "B4", scale_factor=1.0)
        
        self.play(Write(harmonic_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create horizontal bar of cyan blocks
        blocks = VGroup()
        widths = [1.0, 0.5, 0.33, 0.25, 0.20]
        for i, w in enumerate(widths):
            rect = Rectangle(width=w, height=0.4, color=CYAN, fill_opacity=0.6, stroke_width=2)
            if i == 0:
                rect.move_to(ORIGIN)
            else:
                rect.next_to(blocks[-1], RIGHT, buff=0)
            blocks.add(rect)
        
        # Resolved Issue 36: Using place_in_area 'D3' to 'D6' for horizontal layout
        self.place_in_area(blocks, "D3", "D6", scale_factor=0.8)
        
        diverges_label = Text("Diverges", color=RED)
        self.place_at_grid(diverges_label, "E4", scale_factor=0.8)
        
        self.play(FadeIn(blocks, shift=RIGHT))
        
        # Animate growth of the bar
        extra_blocks = VGroup(*[
            Rectangle(width=0.15/(j+1), height=0.4, color=CYAN, fill_opacity=0.6, stroke_width=1)
            for j in range(8)
        ]).arrange(RIGHT, buff=0).next_to(blocks, RIGHT, buff=0)
        
        self.play(Create(extra_blocks), run_time=2)
        self.play(Write(diverges_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Show convergent series (Basel problem)
        basel_formula = MathTex(r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}", color=YELLOW)
        self.place_at_grid(basel_formula, "B4", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(harmonic_formula, basel_formula),
            FadeOut(blocks),
            FadeOut(extra_blocks),
            FadeOut(diverges_label)
        )
        
        # Visualize convergence with shorter blocks
        conv_blocks = VGroup()
        c_widths = [1.0, 0.25, 0.11, 0.06, 0.04, 0.02]
        for i, w in enumerate(c_widths):
            rect = Rectangle(width=w, height=0.4, color=YELLOW, fill_opacity=0.6, stroke_width=2)
            if i == 0:
                rect.move_to(ORIGIN)
            else:
                rect.next_to(conv_blocks[-1], RIGHT, buff=0)
            conv_blocks.add(rect)
            
        self.place_in_area(conv_blocks, "D3", "D6", scale_factor=0.8)
        
        converges_label = Text("Converges", color=GREEN)
        self.place_at_grid(converges_label, "E4", scale_factor=0.8)
        
        self.play(FadeIn(conv_blocks, shift=UP))
        self.play(Write(converges_label))
        self.wait(2)
