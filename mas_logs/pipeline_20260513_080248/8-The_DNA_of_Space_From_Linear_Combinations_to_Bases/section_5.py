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
        title = "The Goldilocks Set: Defining a Basis"
        lines = [
            "A basis is a perfect, efficient toolkit.",
            "It must span the entire target space.",
            "It must also be linearly independent.",
            "No redundant vectors, yet every point reachable.",
            "This minimal set defines the coordinate system."
        ]
        self.setup_layout(title, lines)

        # Colors for lecture highlights
        colors = ["#00FFFF", "#FFFF00", "#FFFFFF", "#FF00FF", "#00FF00"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Vectors and toolkit icon
        # Define vectors relative to a local origin, then move
        origin_pt = ORIGIN
        b1_vec = np.array([1.2, 0.6, 0])
        b2_vec = np.array([0.4, 1.3, 0])
        
        b1 = Arrow(origin_pt, b1_vec, color="#00FFFF", buff=0)
        b2 = Arrow(origin_pt, b2_vec, color="#FFFF00", buff=0)
        b1_lab = Text("b1", font_size=20, color="#00FFFF").next_to(b1.get_end(), RIGHT, buff=0.1)
        b2_lab = Text("b2", font_size=20, color="#FFFF00").next_to(b2.get_end(), UP, buff=0.1)
        
        toolkit_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/toolkit.svg", height=1)
        basis_txt = Text("Basis", font_size=24, color=WHITE)
        toolkit_group = VGroup(toolkit_icon, basis_txt).arrange(DOWN, buff=0.2)
        
        # Positioning
        basis_system = VGroup(b1, b2, b1_lab, b2_lab)
        self.place_in_area(basis_system, "C2", "E4", scale_factor=0.9)
        self.place_at_grid(toolkit_group, "B5", scale_factor=0.8)
        
        self.play(
            FadeIn(basis_system),
            FadeIn(toolkit_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Span demonstration: Sweep out the plane
        span_rect = Rectangle(width=4, height=4, fill_opacity=0.2, fill_color=WHITE, stroke_width=0)
        span_rect.move_to(basis_system[0].get_start()) # Move to system origin
        
        # Create many small arrows to simulate sweep
        sweep_arrows = VGroup()
        for i in range(10):
            scale_b1 = np.random.uniform(-1.5, 1.5)
            scale_b2 = np.random.uniform(-1.5, 1.5)
            end_p = scale_b1 * b1_vec + scale_b2 * b2_vec
            # Shift by system origin (which was moved by place_in_area)
            shift_vec = basis_system[0].get_start()
            sw_arr = Arrow(shift_vec, shift_vec + end_p, buff=0, stroke_width=1, max_tip_length_to_length_ratio=0.1, color=GREY_C)
            sweep_arrows.add(sw_arr)

        self.play(
            Create(span_rect),
            LaggedStart(*[GrowArrow(a) for a in sweep_arrows], lag_ratio=0.1),
            run_time=2
        )
        self.play(FadeOut(sweep_arrows), span_rect.animate.set_fill(opacity=0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Flash vectors to show linear independence
        flash_b1 = b1.copy().set_color(WHITE).set_stroke(width=6)
        flash_b2 = b2.copy().set_color(WHITE).set_stroke(width=6)
        
        self.play(
            FadeIn(flash_b1), FadeIn(flash_b2),
            Flash(b1.get_end(), color=WHITE),
            Flash(b2.get_end(), color=WHITE)
        )
        self.play(FadeOut(flash_b1), FadeOut(flash_b2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # Target Point P and unique combination
        origin_pos = b1.get_start()
        # Pick a target point P
        p_coords = [0.8, 1.0] # coeffs for b1, b2
        p_pos = origin_pos + p_coords[0]*b1.get_vector() + p_coords[1]*b2.get_vector()
        
        dot_p = Dot(p_pos, color="#FF00FF")
        lab_p = Text("P", font_size=20, color="#FF00FF").next_to(dot_p, UR, buff=0.1)
        
        path_b1 = Arrow(origin_pos, origin_pos + p_coords[0]*b1.get_vector(), color="#00FFFF", buff=0, stroke_width=2)
        path_b2 = Arrow(path_b1.get_end(), p_pos, color="#FFFF00", buff=0, stroke_width=2)
        
        self.play(FadeIn(dot_p), Write(lab_p))
        self.play(Create(path_b1))
        self.play(Create(path_b2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        # Transition to standard coordinate grid
        # Create a small grid
        std_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"stroke_opacity": 0.6}
        ).scale(0.8)
        self.place_at_grid(std_grid, "D3") # Center the grid around the vector origin area
        
        # Standard basis i and j
        i_vec = Arrow(std_grid.c2p(0,0), std_grid.c2p(1,0), color="#00FFFF", buff=0)
        j_vec = Arrow(std_grid.c2p(0,0), std_grid.c2p(0,1), color="#FFFF00", buff=0)
        
        self.play(
            FadeOut(span_rect),
            FadeOut(toolkit_group),
            FadeOut(dot_p), FadeOut(lab_p), FadeOut(path_b1), FadeOut(path_b2),
            FadeOut(b1_lab), FadeOut(b2_lab),
            ReplacementTransform(b1, i_vec),
            ReplacementTransform(b2, j_vec),
            Create(std_grid),
            run_time=2
        )
        self.wait(2)
