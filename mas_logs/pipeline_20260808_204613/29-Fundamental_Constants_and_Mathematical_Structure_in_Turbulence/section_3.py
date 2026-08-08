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
        self.setup_layout("Mathematical Structure: The Energy Spectrum", [
            "Energy follows the Kolmogorov -5/3 spectrum.",
            "Wavenumber k relates to eddy scale size.",
            "The spectrum shows power density versus wavenumber.",
            "Energy cascades down across these scales.",
            "This structure reveals underlying mathematical order."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Addressing issue 32: adjust positioning/scale
        formula = MathTex(r"E(k) \propto k^{-5/3}", font_size=36)
        self.place_at_grid(formula, 'B2', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[0].set_color("#F1C40F")

        # === Animation for Lecture Line 2 ===
        # Addressing issue 33: adjust positioning/scale
        axes = Axes(x_range=[0, 2], y_range=[-3, 1], axis_config={"include_tip": False})
        axes.scale(0.5)
        self.place_in_area(axes, 'C2', 'E4', scale_factor=0.5)
        
        graph = axes.plot(lambda k: -1.66 * k, x_range=[0.1, 1.5], color=BLUE)
        self.play(Create(axes), Create(graph))
        self.lecture[1].set_color("#3498DB")

        # === Animation for Lecture Line 3 ===
        # Addressing issue 23 (Asset part 1): Energy cascades
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=WHITE)
        particles = VGroup(*[particle.copy().scale(0.2) for _ in range(5)])
        self.place_in_area(particles, 'C5', 'E6', scale_factor=0.6)
        self.play(FadeIn(particles))
        self.play(particles.animate.arrange(DOWN).shift(RIGHT*0.5))
        self.lecture[2].set_color("#E74C3C")

        # === Animation for Lecture Line 4 ===
        self.play(FadeOut(particles))
        self.lecture[3].set_color("#2ECC71")

        # === Animation for Lecture Line 5 ===
        # Addressing issue 23 (Asset part 2): Grid structure
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color="#2ECC71")
        structured_grid = VGroup(*[grid_icon.copy().scale(0.3) for _ in range(9)])
        structured_grid.arrange_in_grid(3, 3)
        self.place_in_area(structured_grid, 'C3', 'E5', scale_factor=0.5)
        self.play(GrowFromCenter(structured_grid))
        
        self.lecture[4].set_color("#9B59B6")
        self.wait(2)
