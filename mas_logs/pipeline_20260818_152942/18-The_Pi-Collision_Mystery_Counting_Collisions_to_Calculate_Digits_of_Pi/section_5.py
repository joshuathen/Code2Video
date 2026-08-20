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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion & Summary", [
            "Pi hides in physical dynamic systems.",
            "Mass ratios link physics to geometry.",
            "Simple collisions reveal deep mathematical constants."
        ])
        
        # Assets
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        # Table elements
        header_table = VGroup(
            Text("Mass Ratio", font_size=24, color=BLUE),
            Text("Collisions", font_size=24, color=YELLOW)
        ).arrange(RIGHT, buff=0.5)
        
        table = VGroup(
            Text("1", font_size=20, color=WHITE),
            Text("3", font_size=20, color=WHITE),
            Text("100", font_size=20, color=WHITE),
            Text("31", font_size=20, color=WHITE),
            Text("10000", font_size=20, color=WHITE),
            Text("314", font_size=20, color=WHITE)
        ).arrange_in_grid(rows=3, cols=2, buff=0.5)

        # Applying requested placements
        self.place_in_area(header_table, 'C2', 'C5', scale_factor=0.75)
        self.place_in_area(table, 'B2', 'E5', scale_factor=0.7)
        
        pi_text = Text("Pi = 3.1415...", font_size=36, color="#FFD700")
        self.place_at_grid(pi_text, 'D3', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(pendulum, 'A5', scale_factor=0.5)
        self.play(FadeIn(pendulum), FadeIn(header_table))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(table))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(blocks, 'E6', scale_factor=0.5)
        self.play(FadeIn(pi_text), FadeIn(blocks))
        
        self.wait(2)
