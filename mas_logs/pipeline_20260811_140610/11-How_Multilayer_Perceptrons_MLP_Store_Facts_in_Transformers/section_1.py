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
        lecture_lines = [
            "Transformers consist of Attention and MLP blocks.",
            "Attention manages context, like a reading desk.",
            "MLP blocks serve as long-term memory."
        ]
        self.setup_layout("Prerequisite: The Transformer Anatomy", lecture_lines)
        
        # Color constants
        COLOR_MLP = "#3498DB"
        COLOR_LAYER = "#E74C3C"
        COLOR_WEIGHTS = "#2ECC71"
        
        # === Animation for Lecture Line 1 ===
        # Use asset desk.svg as a background for transformer
        desk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/desk.svg")
        transformer_rect = Rectangle(width=3, height=4, color=WHITE)
        attn_box = Rectangle(width=2.5, height=1.5, color=WHITE).next_to(transformer_rect.get_top(), DOWN, buff=0.2)
        mlp_box = Rectangle(width=2.5, height=1.5, color=COLOR_MLP).next_to(attn_box, DOWN, buff=0.2)
        
        transformer_group = VGroup(transformer_rect, attn_box, mlp_box)
        # B4-C6 area for Transformer
        self.place_in_area(transformer_group, "B4", "C6", scale_factor=0.6)
        
        self.play(FadeIn(desk_icon), Create(transformer_rect), Create(attn_box), Create(mlp_box))
        self.play(self.lecture[0].animate.set_color(COLOR_MLP))
        
        # === Animation for Lecture Line 2 ===
        # Illustrate Reading Desk analogy
        self.place_in_area(desk_icon, "E3", "F5", scale_factor=0.8)
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 3 ===
        # Zoom into MLP sub-layer (the bookshelf)
        mlp_internal = VGroup(
            Rectangle(width=2, height=0.5, color=COLOR_LAYER),
            Rectangle(width=2, height=0.5, color=COLOR_LAYER)
        ).arrange(DOWN, buff=0.1)
        
        w1_label = Text("W1", color=COLOR_WEIGHTS, font_size=24)
        w2_label = Text("W2", color=COLOR_WEIGHTS, font_size=24)
        
        # Address critique: Fix overlap with better placement
        self.place_in_area(mlp_internal, "B3", "C5", scale_factor=0.9)
        self.place_at_grid(w1_label, "B6", scale_factor=0.5)
        
        self.play(
            FadeOut(transformer_group),
            Create(mlp_internal),
            Write(w1_label)
        )
        self.play(self.lecture[2].animate.set_color(COLOR_MLP))
        self.wait(2)
