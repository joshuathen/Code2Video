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
        self.setup_layout("Pattern Recognition (The Trap)", [
            "The sequence is 1, 2, 4, 8, 16.",
            "Each step seems to double the last.",
            "Predicting 32 is a natural trap."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mouse.svg
        seq_text = Text("1, 2, 4, 8, 16", color=WHITE, font_size=36)
        self.place_in_area(seq_text, "B4", "B6", scale_factor=0.7)
        
        mouse_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mouse.svg")
        self.place_at_grid(mouse_icon, "A5", scale_factor=0.6)
        
        self.play(Write(seq_text), FadeIn(mouse_icon))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Fix: Line 64: self.place_at_grid(doubling_text, 'C4', scale_factor=0.9)
        self.wait(1)
        doubling_text = Text("x 2", color="#FFD700", font_size=28)
        self.place_at_grid(doubling_text, "C4", scale_factor=0.7) # Using 0.7 for label per B020
        self.play(FadeIn(doubling_text))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Fix: 1. Move predict_text to D3-D5 (scale 0.8)
        # Fix: 2. Move formula to E4 (scale 0.9)
        predict_text = Text("32?", color="#FF4500", font_size=48)
        self.place_in_area(predict_text, "D4", "D6", scale_factor=0.8) # Adjusted area to keep in 4-6
        
        # Assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/trap.svg
        trap_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/trap.svg")
        self.place_at_grid(trap_icon, "E6", scale_factor=0.5)

        formula = MathTex("2^{n-1}", color="#FFFFFF", font_size=32)
        self.place_at_grid(formula, "E4", scale_factor=0.8) # B020: 0.7-0.8 for labels
        
        self.play(GrowFromCenter(predict_text), FadeIn(trap_icon))
        self.play(Write(formula))
        self.lecture[2].set_color("#FF4500")
        
        self.wait(2)
