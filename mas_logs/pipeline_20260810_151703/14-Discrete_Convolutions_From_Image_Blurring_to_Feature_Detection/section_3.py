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
        self.setup_layout("Prerequisite Review: Local Influence", [
            "Each output pixel gathers local influence.",
            "Spatial information flows into synthesized results.",
            "Local neighborhoods determine output characteristics."
        ])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg
        # Represented by a square for now as the asset file might not be accessible during generation, 
        # but code must account for it as instructed.
        
        # 3x3 pixel grid
        pixels = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg") for _ in range(9)])
        pixels.arrange_in_grid(3, 3, buff=0.1)
        
        # Grid placement per feedback: self.place_at_grid(input_grid, 'C4', scale_factor=0.75)
        self.place_at_grid(pixels, 'C4', scale_factor=0.75)
        
        target_pixel = pixels[4].copy().set_color("#ffcc00")
        
        highlight_box = Square(side_length=pixels.width, stroke_color=RED, stroke_width=4)
        highlight_box.move_to(pixels.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ffcc00"))
        self.play(FadeIn(pixels))
        self.play(pixels[4].animate.set_color("#ffcc00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00ffcc"))
        self.play(Create(highlight_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#cc00ff"))
        # Colored glow propagation
        glow = VGroup()
        for i in range(9):
            if i != 4:
                g = pixels[i].copy().set_color("#cc00ff").set_opacity(0.5)
                glow.add(g)
        self.play(FadeIn(glow))
        self.play(glow.animate.shift(target_pixel.get_center() - glow.get_center()))
        self.wait(2)
