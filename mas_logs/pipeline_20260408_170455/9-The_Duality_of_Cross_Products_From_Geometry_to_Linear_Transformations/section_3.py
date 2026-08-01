from manim import *

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
        # Setup layout
        lecture_lines = [
            'Consider a transformation mapping space to a line.',
            'It calculates the signed volume with a third vector.',
            'This linear map corresponds to a dot product.',
            'The cross product acts as the dual vector.',
            'It bridges 3D volumes to 1D scalar outputs.'
        ]
        self.setup_layout("The Linear Transformation Perspective", lecture_lines)

        # Define projection for 2D-looking 3D
        origin = self.grid["D3"]
        basis_i = np.array([0.8, -0.2, 0])
        basis_j = np.array([0.5, 0.4, 0])
        basis_k = np.array([0, 0.7, 0])

        def project(coords):
            x, y, z = coords
            return origin + x * basis_i + y * basis_j + z * basis_k

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Equation f(w) = det(u, v, w)
        equation = Text("f(w) = det(u, v, w)", font_size=24, color=WHITE)
        self.place_in_area(equation, "A1", "A3", scale_factor=0.7) # Fixed Issue 33
        
        # Grid lines (2D-projected 3D grid)
        grid_lines = VGroup()
        for i in range(-1, 2):
            for j in range(-1, 2):
                grid_lines.add(Line(project([i, j, -1]), project([i, j, 1]), color=GRAY, stroke_opacity=0.5))
                grid_lines.add(Line(project([-1, i, j]), project([1, i, j]), color=GRAY, stroke_opacity=0.5))
                grid_lines.add(Line(project([i, -1, j]), project([i, 1, j]), color=GRAY, stroke_opacity=0.5))

        self.play(Write(equation), Create(grid_lines))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        u_val, v_val = [1, 0, 0], [0, 1, 0]
        w_val = [0, 0, 1]
        
        u_vec = Arrow(origin, project(u_val), buff=0, color=WHITE, stroke_width=4)
        v_vec = Arrow(origin, project(v_val), buff=0, color=WHITE, stroke_width=4)
        w_vec = Arrow(origin, project(w_val), buff=0, color=WHITE, stroke_width=4)
        
        w_label = Text("w", font_size=20, color=WHITE)
        w_label.next_to(w_vec.get_end(), UP, buff=0.1)

        def get_box(w_coords):
            points = [
                [0,0,0], [1,0,0], [1,1,0], [0,1,0],
                [0,0,w_coords[2]], [1,0,w_coords[2]], [1,1,w_coords[2]], [0,1,w_coords[2]]
            ]
            proj_pts = [project(p) for p in points]
            edges = VGroup()
            edges.add(Polygon(proj_pts[0], proj_pts[1], proj_pts[2], proj_pts[3], color=GREEN, fill_opacity=0.2, stroke_width=2))
            edges.add(Polygon(proj_pts[4], proj_pts[5], proj_pts[6], proj_pts[7], color=GREEN, fill_opacity=0.2, stroke_width=2))
            for k in range(4):
                edges.add(Line(proj_pts[k], proj_pts[k+4], color=GREEN, stroke_width=2))
            return edges

        box = get_box(w_val)
        self.play(GrowArrow(u_vec), GrowArrow(v_vec), GrowArrow(w_vec), Write(w_label))
        self.play(Create(box))
        
        new_w_val = [0.2, 0.2, 1.2]
        self.play(
            w_vec.animate.put_start_and_end_on(origin, project(new_w_val)),
            w_label.animate.move_to(project(new_w_val) + UP*0.2),
            box.animate.become(get_box(new_w_val)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        number_line = NumberLine(x_range=[-2, 2, 1], length=4, include_tip=True, color=WHITE)
        self.place_in_area(number_line, "F2", "F5", scale_factor=0.8) # Fixed Issue 34
        nl_label = Text("Scalar Output (Line)", font_size=16, color=WHITE).next_to(number_line, DOWN, buff=0.2)
        mapped_dot = Dot(number_line.n2p(new_w_val[2]), color=WHITE)

        self.play(Create(number_line), Write(nl_label))
        
        # Collapse grid and box to origin, map w to line
        self.play(
            FadeOut(grid_lines),
            FadeOut(box),
            FadeOut(u_vec),
            FadeOut(v_vec),
            w_vec.animate.put_start_and_end_on(number_line.n2p(0), number_line.n2p(new_w_val[2])),
            w_label.animate.next_to(number_line.n2p(new_w_val[2]), UP, buff=0.1),
            FadeIn(mapped_dot),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # Re-show the original space lightly
        ghost_grid = grid_lines.copy().set_stroke(opacity=0.1)
        self.add(ghost_grid)
        
        dual_vec_val = [0, 0, 1]
        dual_vec = Arrow(origin, project(dual_vec_val), buff=0, color="#3357FF", stroke_width=6)
        
        dual_label = Text("u × v", font_size=20, color="#3357FF")
        dual_label.next_to(dual_vec.get_end(), LEFT, buff=0.2)
        dual_tag = Text("Dual Vector", font_size=18, color="#3357FF").next_to(dual_label, DOWN, buff=0.1)

        self.play(GrowArrow(dual_vec), Write(dual_label), Write(dual_tag))
        
        dot_eq = Text("f(w) = (u × v) · w", font_size=22, color=WHITE)
        self.place_in_area(dot_eq, "A4", "A6", scale_factor=0.7) # Fixed Issue 32
        self.play(Write(dot_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )

        bridge_line = DashedLine(dual_vec.get_end(), mapped_dot.get_center(), color="#3357FF", stroke_opacity=0.4)
        self.play(Create(bridge_line))
        self.play(Indicate(dot_eq, color=YELLOW), Indicate(number_line, color=YELLOW))
        self.wait(2)
