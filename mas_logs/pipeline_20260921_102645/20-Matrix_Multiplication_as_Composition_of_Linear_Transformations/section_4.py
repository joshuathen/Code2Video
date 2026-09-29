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
        lecture_lines = [
            "Matrix multiplication is non-commutative.",
            "Rotate then stretch differs from stretch then rotate.",
            "Final grid orientation confirms this difference."
        ]
        self.setup_layout("Visualizing Non-Commutativity", lecture_lines)
        
        # Assets
        grid_file = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"
        
        # Mobjects
        grid1 = SVGMobject(grid_file).scale(0.8)
        grid2 = SVGMobject(grid_file).scale(0.8)
        
        # Place grids
        self.place_at_grid(grid1, 'B2', scale_factor=0.8)
        self.place_at_grid(grid2, 'E2', scale_factor=0.8)
        self.add(grid1, grid2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        # Rotate then Stretch (Grid 1)
        self.play(
            Rotate(grid1, angle=PI/4),
            grid1.animate.stretch(1.5, dim=0)
        )
        self.wait(0.5)

        # Stretch then Rotate (Grid 2)
        self.play(
            grid2.animate.stretch(1.5, dim=0),
            Rotate(grid2, angle=PI/4)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(Indicate(grid1), Indicate(grid2))
        self.wait(2)
