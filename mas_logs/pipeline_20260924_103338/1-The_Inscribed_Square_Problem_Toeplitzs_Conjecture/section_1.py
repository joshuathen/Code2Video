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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Can four points always form a square?",
            "A circle makes this easy to see.",
            "But what about a random squiggle?",
            "The Toeplitz conjecture asks this question.",
            "The answer is always yes."
        ]
        self.setup_layout("The Inscribed Square Problem", lecture_lines)
        
        # Load assets
        circle_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        
        # === Animation for Lecture Line 1 ===
        # Fixed: corrected keys to match existing grid definitions (e.g., 'B4' and 'C5' separately)
        points = VGroup(*[Dot(color="#FFFFFF").move_to(self.grid[pos]) for pos in ["B4", "C5", "D4", "E5", "C6", "D6", "B5", "E4"]])
        self.play(FadeIn(points))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_in_area(circle_asset, "C4", "E6", scale_factor=0.6)
        circle_asset.set_color("#00FFFF")
        self.play(ReplacementTransform(points, circle_asset))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        squiggle = VMobject(color="#FFFF00")
        squiggle.set_points_smoothly([self.grid[p] for p in ["C4", "C6", "E6", "E4", "C4"]])
        self.play(ReplacementTransform(circle_asset.copy(), squiggle))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        square = Square(color="#FF0000", fill_opacity=0.3)
        self.place_in_area(square, "C4", "D5", scale_factor=0.4)
        
        # Highlight asset as requested
        self.play(Create(square), circle_asset.animate.set_color("#FF0000"))
        self.lecture[4].set_color("#FF0000")
        self.wait(2)
