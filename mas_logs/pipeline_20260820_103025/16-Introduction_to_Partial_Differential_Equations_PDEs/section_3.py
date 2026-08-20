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
        lecture_lines = [
            "The heat equation links diffusion to spatial curvature.",
            "Rate of temperature change equals the diffusion rate.",
            "Visualize heat spreading from a warm radiator.",
            "[Asset: cat_on_radiator_diffusing]",
            "Curvature drives the flow of heat over time."
        ]
        self.setup_layout("The Core Equation: The Heat Equation", lecture_lines)
        
        # Define objects
        eq = MathTex(r"u_t = \alpha \cdot u_{xx}", font_size=40, color=WHITE)
        # Fix 1: Adjust formula placement
        self.place_in_area(eq, 'B4', 'B6', scale_factor=0.8)
        
        # Assets
        radiator_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radiator.svg", color="#FF4500")
        cat_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")

        # Fix 2 & 3: Align radiator and cat assets
        self.place_at_grid(radiator_img, 'E5', scale_factor=0.5)
        radiator_label = Text("Radiator", font_size=20, color="#FF4500").scale(0.7)
        radiator_label.next_to(radiator_img, DOWN, buff=0.1)

        self.place_at_grid(cat_img, 'C5', scale_factor=0.4)
        cat_label = Text("Target", font_size=20, color="#FFFF00").scale(0.7)
        cat_label.next_to(cat_img, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Write(eq))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF4500")
        self.play(FadeIn(radiator_img), Write(radiator_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        self.play(FadeIn(cat_img), Write(cat_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FF00")
        # Highlight diffusion term u_xx
        u_xx_part = eq[0][6:]
        self.play(Indicate(u_xx_part, color="#00FF00"))
        self.wait(2)
