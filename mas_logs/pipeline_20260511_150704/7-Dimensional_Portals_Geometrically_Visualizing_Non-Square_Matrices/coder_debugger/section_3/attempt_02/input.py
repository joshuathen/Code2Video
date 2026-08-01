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
        # Setup the layout with specific lines
        lines = [
            'A two-by-three matrix squashes three dimensions into two.',
            'Three basis vectors are flattened onto a flat plane.',
            'Like shadow puppetry, we lose the dimension of depth.',
            'Information is lost when projecting into fewer dimensions.',
            'This transformation reduces the volume into an area.'
        ]
        self.setup_layout("Wide Matrices: The Great Squish (3D to 2D)", lines)

        # Pre-define colors for better visual organization
        colors = ["#6495ED", "#32CD32", "#FFD700", "#FF4500", "#DA70D6"]

        # Helper for fake 3D projection (Isometric-like)
        # u_x, u_y, u_z are basis directions in the 2D plane to simulate 3D
        u_x = np.array([1, -0.2, 0])
        u_y = np.array([0.5, 0.5, 0])
        u_z = np.array([0, 1, 0])

        def fake_3d_proj(point):
            return point[0] * u_x + point[1] * u_y + point[2] * u_z

        # === Animation for Lecture Line 1 ===
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Matrix using standard Text to circumvent LaTeX dependency
        matrix = Text("M = [[1, 0.5, 0], [0, 0.5, 1]]", font="Monospace", font_size=24, color=WHITE)
        self.place_in_area(matrix, "A2", "B5", scale_factor=0.9)
        
        # 3D Basis Vectors simulated in 2D
        origin = self.grid["D4"]
        i_vec = Arrow(origin, origin + u_x * 1.5, buff=0, color=BLUE)
        j_vec = Arrow(origin, origin + u_y * 1.5, buff=0, color=GREEN)
        k_vec = Arrow(origin, origin + u_z * 1.5, buff=0, color=RED)
        
        # Using Text for labels to avoid LaTeX dependency
        i_lab = Text("i", slant=ITALIC, color=BLUE, font_size=20).next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_lab = Text("j", slant=ITALIC, color=GREEN, font_size=20).next_to(j_vec.get_end(), UR, buff=0.1)
        k_lab = Text("k", slant=ITALIC, color=RED, font_size=20).next_to(k_vec.get_end(), UP, buff=0.1)

        basis_group = VGroup(i_vec, j_vec, k_vec, i_lab, j_lab, k_lab)
        
        self.play(Write(matrix))
        self.play(Create(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flatten the 3D space onto a 2D plane
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        target_i = Arrow(origin, origin + RIGHT * 1.5, buff=0, color=BLUE)
        target_j = Arrow(origin, origin + (RIGHT + UP) * 0.75, buff=0, color=GREEN)
        target_k = Arrow(origin, origin + UP * 1.5, buff=0, color=RED)
        
        self.play(
            Transform(i_vec, target_i),
            Transform(j_vec, target_j),
            Transform(k_vec, target_k),
            i_lab.animate.next_to(target_i.get_end(), RIGHT),
            j_lab.animate.next_to(target_j.get_end(), UR),
            k_lab.animate.next_to(target_k.get_end(), UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        def get_cube_lines(center, size=0.6):
            s = size
            v = [
                fake_3d_proj(np.array([x, y, z])) 
                for x in [-s, s] for y in [-s, s] for z in [-s, s]
            ]
            edges = [
                (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), 
                (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
            ]