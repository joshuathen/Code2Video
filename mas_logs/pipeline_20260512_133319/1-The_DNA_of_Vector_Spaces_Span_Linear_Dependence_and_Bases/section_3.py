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
        lecture_lines = [
            'Take two vectors pointing in different directions.', 
            'Together, they generate a grid of possible paths.', 
            'Scaling and adding them reaches any point on this grid.', 
            'This entire reachable region is called the span.', 
            'If parallel, the span collapses to a single line.'
        ]
        self.setup_layout("Span: The Reachable Universe", lecture_lines)
        
        # Origin for vector visualization
        origin = self.grid["D4"]
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        v1_vec = np.array([1.2, 0.4, 0])
        v2_vec = np.array([0.4, 1.2, 0])
        
        v1 = Arrow(start=origin, end=origin + v1_vec, color="#00FF00", buff=0, stroke_width=6)
        v2 = Arrow(start=origin, end=origin + v2_vec, color="#0000FF", buff=0, stroke_width=6)
        
        v1_label = Text("v1", font_size=20, color="#00FF00")
        v2_label = Text("v2", font_size=20, color="#0000FF")
        
        v1_label.move_to(origin + v1_vec + [0.4, 0.1, 0])
        v2_label.move_to(origin + v2_vec + [0.1, 0.4, 0])

        self.play(GrowArrow(v1), Write(v1_label))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        grid_lines = VGroup()
        for i in range(-5, 6):
            # Lines parallel to v1 and v2 forming a parallelogram grid
            l1 = Line(origin + i*0.4*v1_vec - 2.5*v2_vec, origin + i*0.4*v1_vec + 2.5*v2_vec, color="#444444", stroke_width=1)
            l2 = Line(origin + i*0.4*v2_vec - 2.5*v1_vec, origin + i*0.4*v2_vec + 2.5*v1_vec, color="#444444", stroke_width=1)
            grid_lines.add(l1, l2)
            
        self.play(Create(grid_lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Show points as linear combinations
        dots = VGroup()
        point_coeffs = [(1.5, 0.5), (-0.8, 1.2), (0.2, -1.5), (-1.2, -0.8), (0.8, 0.8)]
        for c1, c2 in point_coeffs:
            dot = Dot(origin + c1*v1_vec + c2*v2_vec, color=WHITE, radius=0.06)
            dots.add(dot)
            
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Shaded background area
        plane_rect = Rectangle(width=5.5, height=5.5, fill_color="#222222", fill_opacity=0.7, stroke_width=0)
        self.place_in_area(plane_rect, "A1", "F6")
        
        span_tag = Text("Span", font_size=28, color=WHITE)
        self.place_at_grid(span_tag, "A5", scale_factor=0.8)  # RESOLVES ISSUE 32

        dim_info = Text("Dimension = 2", font_size=22, color=WHITE)
        self.place_in_area(dim_info, "F3", "F4", scale_factor=0.9)  # RESOLVES ISSUE 33

        self.play(FadeIn(plane_rect, target_position=origin), Write(span_tag), Write(dim_info))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Transform v2 to overlap v1 (parallel)
        v2_parallel_vec = v1_vec * 0.7
        v2_target_arrow = Arrow(start=origin, end=origin + v2_parallel_vec, color="#0000FF", buff=0, stroke_width=6)
        
        # Area collapses to a line
        collapsed_line = Line(origin - 2.5*v1_vec, origin + 2.5*v1_vec, color="#FFFF00", stroke_width=8)
        
        # Update dimension info
        dim_info_new = Text("Dimension = 1", font_size=22, color="#FFFF00")
        self.place_in_area(dim_info_new, "F3", "F4", scale_factor=0.9)

        self.play(
            Transform(v2, v2_target_arrow),
            v2_label.animate.move_to(origin + v2_parallel_vec + [0.1, -0.4, 0]),
            ReplacementTransform(plane_rect, collapsed_line),
            FadeOut(grid_lines),
            FadeOut(dots),
            span_tag.animate.set_color("#FFFF00").scale(0.8).move_to(self.grid["C6"]),
            Transform(dim_info, dim_info_new)
        )
        self.play(Indicate(dim_info, color=YELLOW), Flash(origin, color=WHITE))
        self.wait(2)
