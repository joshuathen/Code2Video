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
        self.setup_layout("Real-world Applications", [
            "- Convolutions are the foundation of computer vision.",
            "- They enable CNNs to recognize complex objects.",
            "- Apps use them to sharpen images automatically."
        ])
        
        # Load Assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        phone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        self.place_at_grid(camera_icon, 'C4', scale_factor=0.5)
        cnn_layers = VGroup(*[Rectangle(height=1.0, width=0.2, color=WHITE) for _ in range(3)])
        cnn_layers.arrange(RIGHT, buff=0.2).next_to(camera_icon, RIGHT, buff=0.5)
        
        self.play(FadeIn(camera_icon), LaggedStart(*[FadeIn(l) for l in cnn_layers], lag_ratio=0.3))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        self.place_at_grid(phone_icon, 'D4', scale_factor=0.5)
        feature_blocks = VGroup(*[Square(side_length=0.3, color=YELLOW) for _ in range(4)])
        feature_blocks.arrange_in_grid(2, 2, buff=0.1).next_to(phone_icon, RIGHT, buff=0.5)
        
        self.play(FadeIn(phone_icon), FadeIn(feature_blocks))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        img_original = Rectangle(height=1.5, width=1.5, color=GREY, fill_opacity=0.5)
        img_refined = Rectangle(height=1.5, width=1.5, color=WHITE, fill_opacity=0.8)
        self.place_at_grid(img_original, 'E5', scale_factor=0.6)
        
        self.play(FadeIn(img_original))
        self.wait(1)
        self.play(Transform(img_original, img_refined))
        self.wait(2)
