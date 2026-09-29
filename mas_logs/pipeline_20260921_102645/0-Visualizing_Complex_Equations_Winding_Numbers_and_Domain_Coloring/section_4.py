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
        self.setup_layout("Domain Coloring: The Visual Language of Functions", [
            "Color maps phase and magnitude of functions.",
            "Complex outputs turn into navigable terrains.",
            "Zeros appear as black holes in color."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Map grid phase to color spectrum in #00FFFF using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg].
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        grid_phase = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2]).add_coordinates()
        self.place_in_area(grid_phase, 'A3', 'F6', scale_factor=0.4)
        self.place_at_grid(map_icon, "A2", scale_factor=0.5)
        self.play(FadeIn(grid_phase), FadeIn(map_icon))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Scale color saturation by magnitude in #FFFFFF.
        mag_rect = Rectangle(width=2, height=2, color="#FFFFFF", fill_opacity=0.3)
        self.place_at_grid(mag_rect, 'C4', scale_factor=0.7)
        self.play(Create(mag_rect))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Render zeros as stationary black dots in #000000 using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg].
        comp_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        zero_dot = Dot(color="#000000", radius=0.15)
        self.place_at_grid(comp_icon, "F2", scale_factor=0.5)
        self.place_at_grid(zero_dot, 'C4', scale_factor=0.5)
        self.play(FadeIn(comp_icon), FadeIn(zero_dot))
        self.lecture[2].set_color("#FFFFFF") 
        self.wait(2)
