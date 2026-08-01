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
            "Coordinates depend on your chosen perspective or basis.",
            "Meet the Human, using a standard North-East grid.",
            "Meet the Owl, using her tilted branch as reference.",
            "A mouse's location looks different to each observer.",
            "Same physical point, but two different sets of coordinates."
        ]
        self.setup_layout("The Perspective Problem: Two Maps, One Forest", lecture_lines)

        # Colors for reference
        STANDARD_COLOR = BLUE_B
        OWL_COLOR = YELLOW_B
        MOUSE_COLOR = "#00FF00"
        HIGHLIGHT_BLUE = "#0000FF"
        HIGHLIGHT_YELLOW = "#FFFF00"
        TEXT_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(STANDARD_COLOR)
        
        human_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_color": STANDARD_COLOR, "stroke_opacity": 0.4}
        )
        owl_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_color": OWL_COLOR, "stroke_opacity": 0.4}
        ).rotate(30 * DEGREES)
        
        grid_group = VGroup(human_grid, owl_grid)
        self.place_in_area(grid_group, "B2", "E5", scale_factor=0.6)
        
        self.play(Create(human_grid), Create(owl_grid), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(STANDARD_COLOR)
        
        mouse_pos = human_grid.c2p(3, 2)
        mouse_icon = Dot(mouse_pos, color=MOUSE_COLOR, radius=0.1)
        mouse_label = MathTex("(3, 2)", color=MOUSE_COLOR, font_size=28).next_to(mouse_icon, UR, buff=0.1)
        
        human_label = Text("Human's Grid", color=STANDARD_COLOR, font_size=20)
        self.place_at_grid(human_label, "A3", scale_factor=0.8)

        self.play(FadeIn(mouse_icon), Write(mouse_label), FadeIn(human_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(OWL_COLOR)
        
        owl_label = Text("Owl's Grid", color=OWL_COLOR, font_size=20)
        self.place_at_grid(owl_label, "A5", scale_factor=0.8)
        
        # Human unit vectors
        i_vec = Arrow(human_grid.c2p(0,0), human_grid.c2p(1,0), color=HIGHLIGHT_BLUE, buff=0, stroke_width=6)
        j_vec = Arrow(human_grid.c2p(0,0), human_grid.c2p(0,1), color=HIGHLIGHT_BLUE, buff=0, stroke_width=6)
        
        self.play(FadeIn(owl_label))
        self.play(Indicate(i_vec), Indicate(j_vec))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(MOUSE_COLOR)
        
        # Owl basis vectors (rotated)
        u1_vec = Arrow(owl_grid.c2p(0,0), owl_grid.c2p(1,0), color=HIGHLIGHT_YELLOW, buff=0, stroke_width=6)
        u2_vec = Arrow(owl_grid.c2p(0,0), owl_grid.c2p(0,1), color=HIGHLIGHT_YELLOW, buff=0, stroke_width=6)
        
        self.play(Indicate(u1_vec), Indicate(u2_vec))
        
        # Coordinate calculation for (3,2) in 30-deg rotated basis:
        # P_owl = R^T * P_human
        # P_owl = [[cos 30, sin 30], [-sin 30, cos 30]] * [3, 2] = [3.6, 0.23]
        owl_coords_text = MathTex("(3.60, 0.23)_{Owl}", color=OWL_COLOR, font_size=28).next_to(mouse_icon, DR, buff=0.1)
        
        self.play(Write(owl_coords_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(TEXT_COLOR)
        
        info_text = Text("One Point, Two Descriptions", color=WHITE, font_size=24)
        self.place_in_area(info_text, "F2", "F5", scale_factor=0.8)
        
        self.play(
            mouse_icon.animate.scale(2),
            Write(info_text)
        )
        self.wait(2)
