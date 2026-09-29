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
        lecture_lines = [
            "CLIP maps text and images to shared vector space.",
            "Aligned vectors share similar directions, showing semantic similarity.",
            "Cosine similarity measures how close these vectors are."
        ]
        self.setup_layout("CLIP: Mapping Language to Visual Space", lecture_lines)
        
        # Define colors (B039)
        text_color = "#FFFFE0" # Light yellow for text-like elements
        image_color = "#FFFFE0" # Light yellow for image-like elements
        matrix_color = "#FF69B4" # HotPink
        
        # === Animation for Lecture Line 1 ===
        # CLIP maps text and images to shared vector space.
        self.lecture[0].set_color(text_color)
        
        # Using assets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg]
        # and [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg]
        text_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg", color=text_color)
        text_label = Text("Text Space", font_size=18, color=text_color)
        text_group = VGroup(text_icon, text_label).arrange(DOWN)
        
        image_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=image_color)
        image_label = Text("Visual Space", font_size=18, color=image_color)
        image_group = VGroup(image_icon, image_label).arrange(DOWN)
        
        # Apply layout fixes from VideoCritic (24, 26)
        self.place_in_area(text_group, 'A3', 'B5', scale_factor=0.7)
        self.place_in_area(image_group, 'E3', 'F5', scale_factor=0.7)
        
        self.play(FadeIn(text_group), FadeIn(image_group))

        # === Animation for Lecture Line 2 ===
        # Aligned vectors share similar directions, showing semantic similarity.
        self.lecture[1].set_color(text_color)
        
        matrix_rect = Rectangle(color=matrix_color, width=2, height=1).set_fill(matrix_color, opacity=0.2)
        matrix_label = Text("CLIP Embedding", font_size=16, color=WHITE)
        matrix_group = VGroup(matrix_rect, matrix_label)
        
        # Apply layout fix from VideoCritic (25)
        self.place_in_area(matrix_group, 'C3', 'D5', scale_factor=0.75)
        
        self.play(FadeIn(matrix_group))

        # === Animation for Lecture Line 3 ===
        # Cosine similarity measures how close these vectors are.
        self.lecture[2].set_color(text_color)
        
        # Represent mapping with arrows
        arrow1 = Arrow(text_group.get_bottom(), matrix_group.get_top(), color=WHITE)
        arrow2 = Arrow(matrix_group.get_bottom(), image_group.get_top(), color=WHITE)
        
        self.play(Create(arrow1), Create(arrow2))
        self.wait(2)
