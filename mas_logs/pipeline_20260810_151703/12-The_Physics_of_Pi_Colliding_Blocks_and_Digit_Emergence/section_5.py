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
        self.setup_layout("Conclusion and Intuitive Synthesis", [
            "Physics acts as an analog computer.",
            "Conservation mirrors circular geometric symmetry.",
            "Pi emerges from simple motion."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show digital computer simulation [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg] of colliding blocks [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg], color #00FF00.
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#00FF00")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color="#00FF00")
        
        self.place_at_grid(computer, 'B2', scale_factor=0.5)
        self.place_at_grid(blocks, 'B4', scale_factor=0.5)
        
        self.play(FadeIn(computer), FadeIn(blocks))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        # Display rotating circular symmetry overlaying the collision, color #FF00FF.
        circle = Circle(radius=1, color="#FF00FF")
        self.place_at_grid(circle, 'E3', scale_factor=0.8)
        
        self.play(Create(circle))
        self.play(Rotate(circle, angle=2*PI, run_time=2))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 3 ===
        # Final overlay of pi emerging from the movement, color #FFFFFF.
        pi_text = MathTex(r"\pi", color="#FFFFFF")
        self.place_at_grid(pi_text, 'E5', scale_factor=2)
        
        self.play(Write(pi_text))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
