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
        self.setup_layout("Synthesis: The 'Nabla' Operator", [
            "The del operator provides universal tools.", 
            "Dot products yield scalar divergence.", 
            "Cross products define vector curl."
        ])
        
        # Note: Asset path provided was /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Assuming the file exists as named.
        nabla_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=WHITE)
        div_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF0000")
        curl_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#00FF00")
        
        nabla = MathTex(r"\\nabla", font_size=64, color=WHITE)
        div = MathTex(r"\\nabla \\cdot \\mathbf{F}", font_size=48, color="#FF0000")
        curl = MathTex(r"\\nabla \\times \\mathbf{F}", font_size=48, color="#00FF00")
        
        # Combine icons and math
        nabla_group = VGroup(nabla_icon, nabla).arrange(RIGHT, buff=0.2)
        div_group = VGroup(div_icon, div).arrange(RIGHT, buff=0.2)
        curl_group = VGroup(curl_icon, curl).arrange(RIGHT, buff=0.2)
        
        # Apply positioning
        self.place_at_grid(nabla_group, 'B5', scale_factor=1.2)
        self.place_in_area(div_group, 'D4', 'D6', scale_factor=0.9)
        self.place_in_area(curl_group, 'F4', 'F6', scale_factor=0.9)
        
        # Hide initially
        div_group.set_opacity(0)
        curl_group.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(nabla_group))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(div_group))
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(curl_group))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
