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
        lecture_lines = [
            "The cycloid is the true Brachistochrone.",
            "It also features the amazing tautochrone property.",
            "All starting points arrive simultaneously.",
            "Nature favors these elegant mathematical shapes."
        ]
        self.setup_layout("Conclusion: Real-world Synthesis", lecture_lines)
        
        # Load Assets
        coaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")

        # === Animation for Lecture Line 1 ===
        # Show cycloid solution application
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(coaster, 'C4', 'E6', scale_factor=0.6)
        self.play(FadeIn(coaster))

        # === Animation for Lecture Line 2 ===
        # Connect derivation to result
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        tautochrone_dot = Dot(color=YELLOW)
        self.place_at_grid(tautochrone_dot, 'B5', scale_factor=0.5)
        self.play(FadeIn(tautochrone_dot))

        # === Animation for Lecture Line 3 ===
        # Simultaneity point
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 4 ===
        # Final summary
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.place_at_grid(ball, 'E2', scale_factor=0.8)
        self.play(FadeIn(ball))
