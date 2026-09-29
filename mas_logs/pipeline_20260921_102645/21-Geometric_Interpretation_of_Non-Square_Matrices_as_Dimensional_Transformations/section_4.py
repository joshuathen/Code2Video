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
        self.setup_layout("Visualizing the Rank: Why Information is Lost", [
            "Rank is the dimensionality of output.",
            "Basis vectors determine output geometry.",
            "Higher-dimensional objects can be flattened."
        ])
        
        # Define objects for animation
        cube = Cube(side_length=1.5, fill_opacity=0.3, color=BLUE)
        output_line = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color="#FFCCCB", stroke_width=6)
        
        # Assets
        projector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/projector.svg")
        screen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/screen.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")

        # Geometry group for area-based placement
        geometry_group = VGroup(cube, output_line, projector, screen, prism)

        # === Animation for Lecture Line 1 ===
        # Rank is the dimensionality of output.
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(geometry_group, 'B2', 'E5', scale_factor=0.8)
        self.place_at_grid(projector, "B2", scale_factor=0.5)
        self.play(Create(cube), FadeIn(projector), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Basis vectors determine output geometry.
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(screen, "E5", scale_factor=0.5)
        self.place_at_grid(output_line, "D2", scale_factor=0.8)
        self.play(FadeIn(screen), FadeIn(output_line), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Higher-dimensional objects can be flattened.
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(prism, "B5", scale_factor=0.5)
        self.play(FadeIn(prism), run_time=0.5)
        self.play(
            cube.animate.scale([1, 0.05, 0.05]),
            cube.animate.set_color("#FFCCCB"),
            run_time=2
        )
        self.wait(2)
