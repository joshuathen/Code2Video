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
        # Define lecture lines
        lecture_lines = [
            'The "Span" is the set of all reachable points.',
            'Scale vectors infinitely to paint the entire territory.',
            'Two non-parallel vectors span an entire 2D plane.',
            "If vectors point the same way, we're stuck.",
            'They only span a single line in that case.'
        ]
        
        self.setup_layout("The Span: The Reachable Territory", lecture_lines)

        # Colors
        u_color = "#98FB98"  # Pale Green
        v_color = "#FF6347"  # Tomato
        grid_color = "#A9A9A9"  # Grey
        flash_color = "#FFFFFF"

        # Coordinate system setup for right-side visual area (A1-F6)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"stroke_width": 1, "include_tip": False, "color": GREY_E}
        )
        self.place_in_area(axes, "A1", "F6")

        # Vectors
        u_vec = Arrow(axes.c2p(0, 0), axes.c2p(1, 0.5), buff=0, color=u_color)
        v_vec = Arrow(axes.c2p(0, 0), axes.c2p(0.5, 1.5), buff=0, color=v_color)
        
        # Use Text instead of MathTex to avoid 'latex' executable dependency
        u_label = Text("u", color=u_color, font_size=24, slant=ITALIC)
        v_label = Text("v", color=v_color, font_size=24, slant=ITALIC)
        
        # Grid setup
        span_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": grid_color,
                "stroke_width": 1,
                "stroke_opacity": 0.4
            }
        )
        self.place_in_area(span_grid, "A1", "F6")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        u_label.next_to(u_vec.get_end(), RIGHT, buff=0.1)
        v_label.next_to(v_vec.get_end(), UP, buff=0.1)
        
        self.play(GrowArrow(u_vec), Write(u_label))
        self.play(GrowArrow(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Lines extending vectors
        u_line = Line(axes.c2p(-3, -1.5), axes.c2p(3, 1.5), color=u_color, stroke_width=2, stroke_opacity=0.6)
        v_line = Line(axes.c2p(-1, -3), axes.c2p(1, 3), color=v_color, stroke_width=2, stroke_opacity=0.6)

        self.play(Create(u_line), Create(v_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Flash effect
        flash_rect = Rectangle(
            width=5, height=5, 
            fill_color=flash_color, fill_opacity=0.4, 
            stroke_width=0
        )
        self.place_in_area(flash_rect, "A1", "F6")

        self.play(FadeIn(span_grid))
        self.play(FadeIn(flash_rect), run_time=0.3)
        self.play(FadeOut(flash_rect), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # New target for v to make it collinear with u
        new_v_end = axes.c2p(2, 1)
        new_v_line = Line(axes.c2p(-3, -1.5), axes.c2p(3, 1.5), color=v_color, stroke_width=2, stroke_opacity=0.6)
        
        self.play(
            v_vec.animate.put_start_and_end_on(axes.c2p(0, 0), new_v_end),
            v_label.animate.next_to(new_v_end, UR, buff=0.1),
            FadeOut(span_grid),
            FadeOut(v_line),
            ReplacementTransform(v_line.copy(), new_v_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Emphasize the single line
        span_path = Line(axes.c2p(-3, -1.5), axes.c2p(3, 1.5), color=WHITE, stroke_width=4)
        
        self.play(Create(span_path))
        self.play(Indicate(span_path, color=YELLOW))
        self.wait(2)