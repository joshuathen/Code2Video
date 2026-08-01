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
        # Data from storyboard
        title_text = "The Mystery: Where are the Facts?"
        lecture_lines = [
            "Transformers process language through attention and MLP layers.",
            "Attention focuses on grammar and relationships between words.",
            "MLP layers act as the model's long-term factual memory."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show Transformer block with 'Attention' and 'MLP' segments in light gray (#D3D3D3).
        self.play(self.lecture[0].animate.set_color("#D3D3D3"))
        
        attn_box = Rectangle(width=3.0, height=1.0, color="#D3D3D3", fill_opacity=0.2)
        mlp_box = Rectangle(width=3.0, height=1.0, color="#D3D3D3", fill_opacity=0.2)
        
        attn_text = Text("Attention", font_size=20, color="#D3D3D3")
        mlp_text = Text("MLP", font_size=20, color="#D3D3D3")
        
        # Fix from Issue 43: Moving boxes more central
        self.place_in_area(attn_box, "B1", "C4")
        self.place_in_area(attn_text, "B1", "C4")
        self.place_in_area(mlp_box, "D1", "E4")
        self.place_in_area(mlp_text, "D1", "E4")
        
        self.play(
            FadeIn(attn_box), FadeIn(attn_text),
            FadeIn(mlp_box), FadeIn(mlp_text)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Label 'Attention' as 'Grammar' and 'MLP' as 'Facts' in white (#FFFFFF).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        grammar_label = Text("Grammar", font_size=18, color=WHITE)
        facts_label = Text("Facts", font_size=18, color=WHITE)
        
        # Fix from Issue 43: Moving labels slightly left
        self.place_at_grid(grammar_label, "B5")
        self.place_at_grid(facts_label, "D5")
        
        arrow_g = Arrow(start=attn_box.get_right(), end=grammar_label.get_left(), buff=0.1, color=WHITE)
        arrow_f = Arrow(start=mlp_box.get_right(), end=facts_label.get_left(), buff=0.1, color=WHITE)
        
        self.play(
            Create(arrow_g), Write(grammar_label),
            Create(arrow_f), Write(facts_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Animate 'MLP' glowing gold (#FFD700) while 'Paris' [Asset: ...] appears in cyan (#00FFFF).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Issue 25: Integrate SVG asset
        paris_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg")
        paris_asset.set_color("#00FFFF")
        
        # Fix from Issue 44: Move Paris to E5
        self.place_at_grid(paris_asset, "E5", scale_factor=0.6)
        
        self.play(
            mlp_box.animate.set_fill("#FFD700", opacity=0.6).set_stroke("#FFD700"),
            mlp_text.animate.set_color("#FFD700"),
            Indicate(mlp_box, color="#FFD700"),
            FadeIn(paris_asset, shift=UP)
        )
        self.wait(3)
