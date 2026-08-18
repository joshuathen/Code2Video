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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Substitute n = c/v into the ratio.", "This derives Snell's Law.", "Predicts light path between media."]
        self.setup_layout("Conclusion: Snell's Law", lecture_lines)
        
        # Load Assets
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        # Elements
        snell_law = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", color=WHITE)
        self.place_in_area(snell_law, 'B2', 'B5', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        # Display Snell's Law and prism icon
        self.place_at_grid(prism, 'C3', scale_factor=0.5)
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Write(snell_law), FadeIn(prism))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), snell_law.animate.set_color("#FFFF00"))
        
        # Highlight n sin(theta) part
        highlight_box = SurroundingRectangle(snell_law, color="#00FF00", buff=0.1)
        self.play(Create(highlight_box))
        self.wait(1)
        self.play(FadeOut(highlight_box))

        # === Animation for Lecture Line 3 ===
        # Finalize derivation and glass icon
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(glass, 'E3', scale_factor=0.5)
        self.play(FadeIn(glass))
        
        # Visual representation: Laser reflection
        line1 = Line(start=np.array([-1, 1, 0]), end=np.array([0, 0, 0]), color=BLUE)
        line2 = Line(start=np.array([0, 0, 0]), end=np.array([1, -0.5, 0]), color=RED)
        boundary = Line(start=np.array([-2, 0, 0]), end=np.array([2, 0, 0]), color=WHITE)
        
        visual = VGroup(line1, line2, boundary)
        self.place_at_grid(visual, 'D3', scale_factor=0.75)
        
        self.play(Create(visual))
        self.wait(2)
