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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Quantum States: The Bloch Sphere", [
            "The Bloch sphere maps these quantum states.",
            "Complex coefficients define points on the surface.",
            "Superposition is a rotation in this space."
        ])
        
        # --- Load Assets ---
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        # --- Create objects ---
        label_0 = Text("|0⟩", color="#00FF00", font_size=20)
        label_1 = Text("|1⟩", color="#00FF00", font_size=20)
        psi_symbol = Text("ψ", color=YELLOW, font_size=20)
        
        state_point = Dot(color=YELLOW)
        state_group = VGroup(state_point, psi_symbol)
        
        # Applying requested layout fixes
        self.place_in_area(sphere_asset, 'B4', 'E6', scale_factor=0.4)
        self.place_at_grid(label_0, 'A4', scale_factor=0.6)
        self.place_at_grid(label_1, 'F4', scale_factor=0.6)
        self.place_at_grid(psi_symbol, 'C3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(sphere_asset), Write(label_0), Write(label_1))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(state_group))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        path = Arc(radius=0.5, start_angle=PI/4, angle=PI/2, color=RED)
        self.play(Rotate(state_group, angle=PI/2, about_point=sphere_asset.get_center()), Create(path))
        self.wait(1)
