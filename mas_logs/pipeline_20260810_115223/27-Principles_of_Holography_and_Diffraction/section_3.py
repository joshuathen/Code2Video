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
        self.setup_layout("The Holographic Process (Recording)", [
            "Holography captures both light amplitude and phase.",
            "Reference and object beams meet on film.",
            "Interference patterns are frozen as a grating."
        ])
        
        # Define elements
        beam1 = Line(start=np.array([-0.5, 0.5, 0]), end=np.array([0, 0, 0]), color=WHITE)
        beam2 = Line(start=np.array([-0.5, -0.5, 0]), end=np.array([0, 0, 0]), color=WHITE)
        beams = VGroup(beam1, beam2)
        
        # Asset Loading
        film_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg")
        film = Rectangle(height=1.0, width=0.1, color=BLUE).set_fill(BLUE, opacity=0.3)
        film_group = VGroup(film, film_asset)
        
        fringes = VGroup(*[
            Line(start=np.array([0, i*0.1, 0]), end=np.array([0.3, i*0.1, 0]), color=YELLOW)
            for i in range(-5, 6)
        ])
        
        grating = Square(side_length=1.5, color=PURPLE).set_fill(PURPLE, opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(Create(beams), FadeIn(film_asset))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Addressing issues 27, 28, 42: Move beams and film, scale them down
        self.place_at_grid(beams, "B4", scale_factor=0.7)
        self.place_at_grid(film_group, "B5", scale_factor=0.7)
        self.play(Create(fringes.move_to(self.grid["B5"])))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Addressing issues 26, 41: Adjust grating position
        self.place_in_area(grating, "D3", "F5", scale_factor=0.5)
        self.play(FadeIn(grating))
        self.lecture[2].set_color("#FF00FF")
        
        # Final summary element
        self.place_at_grid(film_asset.copy(), "F6", scale_factor=0.5)
        
        self.wait(2)
