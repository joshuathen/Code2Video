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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            'We define grids using standard basis vectors i and j.',
            'Every vector is a combination of these unit vectors.',
            'This standard perspective gives us familiar coordinates like [3, 2].'
        ]
        self.setup_layout("Prerequisite: The Standard Basis", lines)

        # Pre-define colors
        COLOR_I = "#00FF00"
        COLOR_J = "#0000FF"
        COLOR_GRID = "#555555"
        COLOR_V = "#FFFFFF"

        # Initialize Grid
        # Adjusting ranges to fill the area nicely
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "include_tip": True}
        )
        # Issue 29 Fix: Use larger area
        self.place_in_area(plane, 'A2', 'F6', scale_factor=1.0)

        # Basis vectors
        i_vec = Arrow(plane.c2p(0,0), plane.c2p(1,0), buff=0, color=COLOR_I)
        j_vec = Arrow(plane.c2p(0,0), plane.c2p(0,1), buff=0, color=COLOR_J)
        
        # Grid-based labels to avoid manual positioning
        i_label = Text("i", slant=ITALIC, color=COLOR_I, font_size=24)
        self.place_at_grid(i_label, 'E3', scale_factor=0.8)
        
        # Issue 30 Fix: j_label at D1
        j_label = Text("j", slant=ITALIC, color=COLOR_J, font_size=24)
        self.place_at_grid(j_label, 'D1', scale_factor=0.8)

        # Vector V components
        step_i = Arrow(plane.c2p(0,0), plane.c2p(3,0), buff=0, color=COLOR_I, stroke_width=2)
        step_j = Arrow(plane.c2p(3,0), plane.c2p(3,2), buff=0, color=COLOR_J, stroke_width=2)
        vector_v = Arrow(plane.c2p(0,0), plane.c2p(3,2), buff=0, color=COLOR_V)
        
        # Issue 31 Fix: coords_label at B6
        coords_label = Text("[3, 2]", color=COLOR_V, font_size=24)
        self.place_at_grid(coords_label, 'B6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(plane), run_time=1.5)
        self.play(
            GrowArrow(i_vec),
            GrowArrow(j_vec),
            Write(i_label),
            Write(j_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Show vector sum components
        self.play(Create(step_i), run_time=1)
        self.play(Create(step_j), run_time=1)
        # Show resulting vector
        self.play(GrowArrow(vector_v), run_time=1)
        self.play(FadeOut(step_i), FadeOut(step_j), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Write(coords_label), run_time=1)
        self.wait(2)

        # Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
