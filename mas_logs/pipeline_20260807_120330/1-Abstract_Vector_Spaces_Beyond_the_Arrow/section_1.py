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
        title = "The Starting Point: Familiar Euclidean Space"
        lines = [
            "Vectors are often pictured as arrows in space.",
            "We add them by placing them tip-to-tail.",
            "We scale them by stretching or shrinking.",
            "These operations define the geometry of R-squared.",
            "But vectors are much more than just arrows."
        ]
        self.setup_layout(title, lines)

        # Assets
        # Requirement [B001, B021, Issue 25]: Load the arrow asset for vector representation
        ARROW_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg"

        # Colors
        GREEN = "#00FF00"
        BLUE = "#00BFFF"
        PURPLE = "#9B30FF"
        YELLOW_C = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a coordinate system in the grid area
        # Occupying C3-D4 area as the visual center
        plane_center = self.grid["C3"] + (self.grid["D4"] - self.grid["C3"]) / 2
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "tip_width": 0.15, "tip_height": 0.15},
            background_line_style={"stroke_opacity": 0.4}
        ).move_to(plane_center)
        
        # Function to generate SVG-based vectors consistent with coordinates
        def get_svg_vec(coords, color):
            start = plane.c2p(0, 0)
            end = plane.c2p(*coords)
            line = Line(start, end)
            length = line.get_length()
            angle = line.get_angle()
            
            vec = SVGMobject(ARROW_ASSET).set_color(color)
            if vec.width > 0:
                vec.scale(length / vec.width)
            
            # Anchor at tail and rotate to target coordinate
            vec.rotate(angle, about_point=vec.get_left())
            vec.shift(start - vec.get_left())
            return vec

        # Requirement [Issue 25]: Use SVG for vector u
        vec_u = get_svg_vec([2, 1], GREEN)
        label_u = MathTex(r"\vec{u}", color=GREEN, font_size=24)
        
        # Requirement [Issue 29]: Position label_u at B5 for proximity
        self.place_at_grid(label_u, "B5")

        self.play(Create(plane))
        self.play(Create(vec_u), Write(label_u))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Requirement [Issue 25]: Vector v as SVG
        vec_v = get_svg_vec([1, 2], BLUE)
        # Tip-to-tail placement: tail of v shifted to tip of u
        vec_v.shift(plane.c2p(2, 1) - plane.c2p(0, 0))
        
        label_v = MathTex(r"\vec{v}", color=BLUE, font_size=24)
        # Requirement [Issue 27]: Adjust label_v to A5
        self.place_at_grid(label_v, "A5")
        
        # Resultant vector u + v
        vec_sum = get_svg_vec([3, 3], PURPLE)
        label_uv = MathTex(r"\vec{u} + \vec{v}", color=PURPLE, font_size=24)
        # Requirement [Issue 27]: Position label_uv at A6
        self.place_at_grid(label_uv, "A6")

        self.play(Create(vec_v), Write(label_v))
        self.wait(0.5)
        self.play(Create(vec_sum), Write(label_uv))
        self.wait(1)

        # Cleanup for next stage
        self.play(FadeOut(vec_v), FadeOut(label_v), FadeOut(vec_sum), FadeOut(label_uv))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Scaling vector u by factor of 2
        # Performance: animate scale directly rather than redrawing SVG every frame
        self.play(vec_u.animate.scale(2, about_point=plane.c2p(0, 0)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight the R^2 coordinate system
        plane_highlight = plane.copy().set_stroke(YELLOW_C, opacity=0.8)
        label_r2 = MathTex(r"\mathbb{R}^2", color=YELLOW_C, font_size=36)
        
        # Requirement [Issue 28]: Move label_r2 from A6 to A5
        self.place_at_grid(label_r2, "A5")

        self.play(Create(plane_highlight), Write(label_r2))
        self.wait(1)
        self.play(FadeOut(plane_highlight))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Abstract away from arrows to prepare for generalized vector spaces
        self.play(
            FadeOut(vec_u),
            FadeOut(label_u),
            FadeOut(label_r2),
            plane.animate.set_stroke(opacity=0.1)
        )
        self.wait(2)

        # Reset last line
        self.lecture[4].set_color(WHITE)
