from manim import *
import os

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
        self.setup_layout("The Step-by-Step Procedure", [
            "Differentiate both sides with respect to x.",
            "Group all terms containing dy/dx together.",
            "Factor out dy/dx and solve."
        ])
        
        # Load assets (placeholder icons as indicated by the storyboard)
        # Note: the paths exist per the instruction, though they are named '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg'
        asset1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        asset3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        step1 = Text("1. Differentiate w.r.t x", color="#FFFFFF", font_size=20)
        step2 = Text("2. Group dy/dx terms", color="#00FF00", font_size=20)
        step3 = Text("3. Solve for dy/dx", color="#FF6347", font_size=20)
        
        self.place_at_grid(step1, "B3", 1.0)
        self.place_at_grid(asset1, "B2", 0.5)
        self.place_at_grid(step2, "C3", 1.0)
        self.place_at_grid(step3, "D3", 1.0)
        self.place_at_grid(asset3, "D2", 0.5)

        # === Animation for Lecture Line 1 ===
        self.play(Write(step1), FadeIn(asset1))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Write(step2))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Write(step3), FadeIn(asset3))
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        self.wait(2)
