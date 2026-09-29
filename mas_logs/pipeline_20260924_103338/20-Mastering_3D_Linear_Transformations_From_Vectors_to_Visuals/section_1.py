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

class Section1Scene(TeachingScene, ThreeDScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: The Basis of 3D Space", [
            "3D vectors live in space.",
            "Basis vectors define axes: i, j, k.",
            "Any vector is their combination."
        ])
        
        # 3D Scene setup
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.6)
        
        # Use SVG for cube
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#FF00FF")
        
        i_hat = Arrow3D(start=ORIGIN, end=RIGHT, color=RED)
        j_hat = Arrow3D(start=ORIGIN, end=UP, color=GREEN)
        k_hat = Arrow3D(start=ORIGIN, end=OUT, color=BLUE)
        
        basis_group = VGroup(i_hat, j_hat, k_hat)
        self.place_at_grid(basis_group, 'B4', scale_factor=0.7)
        
        i_label = MathTex(r"\\hat{i}", color=RED).next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label = MathTex(r"\\hat{j}", color=GREEN).next_to(j_hat.get_end(), UP, buff=0.1)
        k_label = MathTex(r"\\hat{k}", color=BLUE).next_to(k_hat.get_end(), OUT, buff=0.1)
        labels = VGroup(i_label, j_label, k_label)

        # Asset for combination
        vector_combination_animation = VGroup(cube) # Placeholder for the combination logic
        self.place_in_area(vector_combination_animation, 'B3', 'C6', scale_factor=0.65)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.add(basis_group, labels)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(cube))
        self.wait(2)
