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
        # Configuration
        title_text = "The Basis: The Efficient Blueprint"
        lecture_lines = [
            "A basis is a minimalist toolkit of unit vectors.",
            "They are linearly independent, meaning no redundant directions.",
            "Together, they span and define the entire coordinate space."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors and Assets
        COLOR_I = "#F0E68C"
        COLOR_J = "#E6E6FA"
        COLOR_GRID = "#333333"
        COLOR_BASIS = "#FFD700"
        BLUEPRINT_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/blueprint.svg"
        TOOLKIT_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/toolkit.svg"

        # === Animation for Lecture Line 1 ===
        # A basis is a minimalist toolkit of unit vectors.
        self.play(self.lecture[0].animate.set_color(COLOR_I))

        # Coordinate System Setup
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )
        
        # Unit Vectors i and j
        vec_i = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            buff=0, color=COLOR_I, stroke_width=6
        )
        vec_j = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(0, 1),
            buff=0, color=COLOR_J, stroke_width=6
        )
        
        coordinate_system_group = VGroup(plane, vec_i, vec_j)
        self.place_in_area(coordinate_system_group, 'A2', 'E5', scale_factor=1.0)
        
        # Blueprint Asset
        blueprint = SVGMobject(BLUEPRINT_PATH)
        self.place_at_grid(blueprint, 'A6', scale_factor=0.6)

        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(vec_i), GrowArrow(vec_j), FadeIn(blueprint))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They are linearly independent, meaning no redundant directions.
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Pulse i and j in white
        self.play(
            vec_i.animate.set_color(WHITE),
            vec_j.animate.set_color(WHITE),
            rate_func=there_and_back,
            run_time=1
        )
        self.play(
            vec_i.animate.set_color(COLOR_I),
            vec_j.animate.set_color(COLOR_J)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Together, they span and define the entire coordinate space.
        self.play(self.lecture[2].animate.set_color(COLOR_BASIS))

        # Target point and scaling logic
        target_coords = (2, 1)
        target_point_abs = plane.coords_to_point(*target_coords)
        target_dot = Dot(target_point_abs, color=WHITE)
        
        # Visualizing the span to reach target point
        vec_i_scaled = Arrow(
            plane.coords_to_point(0,0), plane.coords_to_point(2,0),
            buff=0, color=COLOR_I, stroke_width=4, opacity=0.6
        )
        vec_j_scaled = Arrow(
            plane.coords_to_point(2,0), plane.coords_to_point(2,1),
            buff=0, color=COLOR_J, stroke_width=4, opacity=0.6
        )

        # Labels anchored to grid
        formula_label = Text("2v1 + 1v2", font_size=24, color=WHITE)
        self.place_at_grid(formula_label, 'B5', scale_factor=0.6)
        
        span_label = Text("Span{v1, v2}", font_size=24, color=COLOR_GRID)
        self.place_at_grid(span_label, 'F3', scale_factor=0.8)

        # Show scaling and coverage
        self.play(
            Create(vec_i_scaled),
            Create(vec_j_scaled),
            FadeIn(target_dot),
            Write(formula_label),
            Write(span_label)
        )
        
        # Basis Toolkit integration
        toolkit = SVGMobject(TOOLKIT_PATH)
        self.place_at_grid(toolkit, 'E6', scale_factor=0.6)
        
        basis_tag = Text("Basis", font_size=32, color=COLOR_BASIS)
        self.place_at_grid(basis_tag, 'D6', scale_factor=0.8)

        self.play(
            FadeIn(toolkit),
            Write(basis_tag),
            Flash(basis_tag, color=COLOR_BASIS)
        )
        
        self.wait(3)
