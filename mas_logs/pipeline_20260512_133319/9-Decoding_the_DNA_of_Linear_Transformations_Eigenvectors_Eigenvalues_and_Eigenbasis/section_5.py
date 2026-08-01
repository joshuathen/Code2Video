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
        # Setup layout
        title_text = "The Eigenbasis: A Better Point of View"
        lines = [
            "Eigenvectors can form a natural coordinate system.",
            "In this Eigenbasis, transformations only scale vectors.",
            "This turns complex matrices into simple diagonal ones.",
            "We decompose the matrix using these fundamental directions.",
            "Now, calculating high matrix powers becomes instant scaling."
        ]
        self.setup_layout(title_text, lines)

        # Colors
        GRID_COLOR = "#52C41A"
        VECTOR_COLOR = "#FAAD14"
        POWER_COLOR = "#F5222D"
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GRID_COLOR)
        
        # Directions for eigenvectors
        v1_dir = np.array([1.0, 0.4, 0])
        v2_dir = np.array([-0.3, 0.8, 0])
        
        # Create tilted grid lines
        tilted_grid = VGroup()
        for i in range(-3, 4):
            # Lines parallel to v1
            start1 = i * v2_dir - 2.5 * v1_dir
            end1 = i * v2_dir + 2.5 * v1_dir
            line1 = Line(start1, end1, stroke_width=1, stroke_opacity=0.4, color=GRID_COLOR)
            tilted_grid.add(line1)
            
            # Lines parallel to v2
            start2 = i * v1_dir - 2.5 * v2_dir
            end2 = i * v1_dir + 2.5 * v2_dir
            line2 = Line(start2, end2, stroke_width=1, stroke_opacity=0.4, color=GRID_COLOR)
            tilted_grid.add(line2)

        # Place the coordinate system center
        visual_anchor = VGroup(tilted_grid)
        self.place_in_area(visual_anchor, "B2", "E6", scale_factor=0.8)
        origin_point = visual_anchor.get_center()

        # Basis arrows
        basis_v1 = Arrow(origin_point, origin_point + v1_dir, color=GRID_COLOR, buff=0)
        basis_v2 = Arrow(origin_point, origin_point + v2_dir, color=GRID_COLOR, buff=0)
        label_v1 = Text("v\u2081", font_size=18, color=GRID_COLOR).next_to(basis_v1.get_end(), RIGHT, buff=0.1)
        label_v2 = Text("v\u2082", font_size=18, color=GRID_COLOR).next_to(basis_v2.get_end(), UP, buff=0.1)

        self.play(FadeIn(tilted_grid))
        self.play(GrowArrow(basis_v1), GrowArrow(basis_v2), Write(label_v1), Write(label_v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(VECTOR_COLOR)
        
        # Vector w in the eigenbasis
        vec_w = Arrow(origin_point, origin_point + v1_dir*0.6 + v2_dir*0.6, color=VECTOR_COLOR, buff=0)
        label_w = Text("w", font_size=18, color=VECTOR_COLOR).next_to(vec_w.get_end(), UR, buff=0.05)
        
        self.play(GrowArrow(vec_w), Write(label_w))
        self.wait(1)
        
        # Transform vector w by scaling along the basis directions
        vec_w_transformed = Arrow(origin_point, origin_point + v1_dir*0.9 + v2_dir*0.3, color=VECTOR_COLOR, buff=0)
        
        self.play(
            ReplacementTransform(vec_w, vec_w_transformed),
            label_w.animate.next_to(vec_w_transformed.get_end(), UR, buff=0.05)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(TEXT_COLOR)
        
        # Diagonal matrix D
        d_label = Text("D = ", font_size=24, color=TEXT_COLOR)
        d_mat = VGroup(
            Text("[", font_size=40, color=TEXT_COLOR),
            VGroup(
                VGroup(Text("\u03BB\u2081", color=TEXT_COLOR), Text("0", color=TEXT_COLOR)).arrange(RIGHT, buff=0.6),
                VGroup(Text("0", color=TEXT_COLOR), Text("\u03BB\u2082", color=TEXT_COLOR)).arrange(RIGHT, buff=0.6)
            ).arrange(DOWN, buff=0.3),
            Text("]", font_size=40, color=TEXT_COLOR)
        ).arrange(RIGHT, buff=0.1)
        d_group = VGroup(d_label, d_mat).arrange(RIGHT, buff=0.2)
        
        # Position D matrix in the top area (Issue 36 fix)
        self.place_in_area(d_group, "A3", "A5", scale_factor=0.8)
        
        self.play(FadeIn(d_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(TEXT_COLOR)
        
        # Decomposition equation A = PDP⁻¹
        decomp_eq = VGroup(
            Text("A = P ", color=TEXT_COLOR),
            Text("D", color=POWER_COLOR), # Highlight D in red
            Text(" P", color=TEXT_COLOR),
            Text("-1", font_size=16, color=TEXT_COLOR).shift(UP*0.2)
        ).arrange(RIGHT, buff=0.05)
        
        # Position decomposition equation at the bottom right (Issue 37 fix)
        self.place_at_grid(decomp_eq, "F5", scale_factor=1.0)
        
        self.play(Write(decomp_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(POWER_COLOR)
        
        # Matrix Power A¹⁰
        power_eq = Text("A\u00B9\u2070", font_size=28, color=POWER_COLOR)
        # Position power equation (Issue 35 fix)
        self.place_at_grid(power_eq, "F3", scale_factor=1.0)
        
        # Extreme scaling of basis vectors representing high powers
        basis_v1_long = Arrow(origin_point, origin_point + v1_dir*2.0, color=POWER_COLOR, buff=0)
        basis_v2_short = Arrow(origin_point, origin_point + v2_dir*0.2, color=POWER_COLOR, buff=0)
        
        self.play(Write(power_eq))
        self.play(
            ReplacementTransform(basis_v1, basis_v1_long),
            ReplacementTransform(basis_v2, basis_v2_short),
            FadeOut(vec_w_transformed),
            FadeOut(label_w)
        )
        self.wait(2)
