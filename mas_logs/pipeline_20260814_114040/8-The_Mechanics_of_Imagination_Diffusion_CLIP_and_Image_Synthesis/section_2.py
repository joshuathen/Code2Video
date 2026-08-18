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
        lecture_lines = ["CLIP bridges the gap between text and images.", "It maps them into a shared vector space.", "Words and images become mathematically close points."]
        self.setup_layout("CLIP: The Linguistic Bridge", lecture_lines)
        
        # Colors per B039
        COLOR_TEXT = "#FF6347"
        COLOR_CONNECT = "#00FF00"
        COLOR_SPACE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show text and image input nodes using assets
        text_label = Text("Cat", font_size=24, color=COLOR_TEXT)
        img_node = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        img_node.set_color(COLOR_TEXT)
        
        # Fixing layout per issues 29, 31
        self.place_at_grid(text_label, "B3", scale_factor=0.8)
        self.place_at_grid(img_node, "B5", scale_factor=0.8)
        
        self.play(FadeIn(text_label), FadeIn(img_node), 
                  self.lecture[0].animate.set_color(COLOR_TEXT))

        # === Animation for Lecture Line 2 ===
        # Animate connection lines using assets
        conn_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        conn_asset.set_color(COLOR_CONNECT)
        self.place_in_area(conn_asset, "C3", "C4", scale_factor=0.8)
        
        line = Line(text_label.get_bottom(), conn_asset.get_top(), color=COLOR_CONNECT)
        line2 = Line(conn_asset.get_top(), img_node.get_bottom(), color=COLOR_CONNECT)
        
        self.play(FadeIn(conn_asset), Create(line), Create(line2), self.lecture[1].animate.set_color(COLOR_CONNECT))

        # === Animation for Lecture Line 3 ===
        # Highlight joint embedding space - fixing layout per issue 30
        space = Rectangle(width=2.5, height=1.5, color=COLOR_SPACE, fill_opacity=0.1)
        self.place_in_area(space, "C3", "D4", scale_factor=0.8)
        
        self.play(FadeIn(space), self.lecture[2].animate.set_color(COLOR_SPACE))
        self.wait(2)
