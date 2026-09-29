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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: High-Dimensional Vector Spaces", [
            "Words and pixels live in vector spaces.",
            "Proximity in these spaces denotes semantic meaning.",
            "This enables math to relate concepts accurately."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Load assets
        word_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/words.svg")
        pixel_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixels.svg")
        
        # Display vector space (using icons as points)
        pt1 = word_icon.copy()
        pt2 = word_icon.copy()
        self.place_at_grid(pt1, 'A2', scale_factor=0.3)
        self.place_at_grid(pt2, 'B5', scale_factor=0.3)
        
        vector_arrow = Arrow(start=pt1.get_center(), end=pt2.get_center(), color="#33FF57")
        scene_objects = VGroup(pt1, pt2, vector_arrow)
        
        self.place_in_area(scene_objects, 'A2', 'B5', scale_factor=0.5)
        self.place_at_grid(vector_arrow, 'A4', scale_factor=0.7)
        
        self.play(Create(pt1), Create(pt2), Create(vector_arrow))
        self.lecture[0].set_color("#33FF57")

        # === Animation for Lecture Line 2 ===
        # Dot cloud representing space
        dot_cloud = VGroup(*[Dot(radius=0.03, color=BLUE) for _ in range(30)])
        for dot in dot_cloud:
            dot.move_to(np.array([float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1)), 0.0]))
        self.place_in_area(dot_cloud, 'D2', 'F5', scale_factor=0.8)
        
        self.play(FadeIn(dot_cloud))
        self.lecture[1].set_color("#FFFF00") # Highlighting proximity

        # === Animation for Lecture Line 3 ===
        # Show math equation relating concepts
        eq = MathTex(r"sim(\vec{v}_{word}, \vec{v}_{pixel})", color=WHITE)
        self.place_in_area(eq, 'C2', 'C5', scale_factor=1.0)
        
        self.play(Write(eq))
        self.lecture[2].set_color(WHITE)
        self.wait(1)
