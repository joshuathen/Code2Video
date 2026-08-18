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
        self.setup_layout("Kolmogorov’s 1941 Hypothesis", [
            "Kolmogorov hypothesized universal statistical behavior.",
            "In the inertial range, energy scales predictably.",
            "This creates the famous -5/3 energy spectrum law.",
            "Large vortices break into smaller, self-similar structures.",
            "Statistical patterns remain independent of the fluid."
        ])
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[0.1, 4], y_range=[0.1, 4], x_length=4, y_length=4)
        graph = axes.plot(lambda x: x**(-5/3), color=WHITE)
        # Fix for Issue 26: Axes overlap
        self.place_in_area(axes, 'A4', 'D6', scale_factor=0.5)
        self.add(axes, graph)
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix for Issue 27: Inertial box obstruction
        inertial_box = Rectangle(width=1.5, height=1, color="#00FF00", fill_opacity=0.2)
        self.place_in_area(inertial_box, 'E2', 'F3', scale_factor=0.6)
        self.play(Create(inertial_box))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix for Issue 28: Spectrum label positioning
        spectrum_label = MathTex("E(k) \\propto k^{-5/3}", color="#FF0000")
        self.place_at_grid(spectrum_label, 'D5', scale_factor=0.7)
        self.play(Write(spectrum_label))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Added Asset per Issue 18
        vortex_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg", color="#FFFF00")
        self.place_at_grid(vortex_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(vortex_icon))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.wait(2)
