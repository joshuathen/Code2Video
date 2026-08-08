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
        # Setup with storyboard lines
        self.setup_layout("The Span: Reachable Territory", [
            "The Span is every point Vector-Bot can possibly reach.",
            "One vector spans a single line of infinite points.",
            "Two non-parallel vectors can \"paint\" an entire 2D plane.",
            "Think of Span as the full territory covered.",
            "If vectors align, the Span collapses to a line."
        ])
        
        # Colors
        SPAN_COLOR = "#FFFF00"
        LINE_COLOR = "#808080"
        PLANE_COLOR = "#ADD8E6"
        V_COLOR = "#00FF00" # Light Green
        W_COLOR = "#FF00FF" # Magenta
        COLLAPSE_COLOR = "#FF3333" # Reddish for collapse

        # === Animation for Lecture Line 1 ===
        # "The Span is every point Vector-Bot can possibly reach."
        self.lecture[0].set_color(SPAN_COLOR)
        span_word = Text("SPAN", weight=BOLD, color=SPAN_COLOR)
        # Resolved Issue 23: adjusted area and scale
        self.place_in_area(span_word, 'B2', 'E5', scale_factor=1.1)
        self.play(Write(span_word))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "One vector spans a single line of infinite points."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(V_COLOR),
            FadeOut(span_word)
        )
        
        # origin at C3, vector towards C5
        origin = self.grid["C3"]
        v_target = self.grid["C5"]
        v_vec = Arrow(origin, v_target, buff=0, color=V_COLOR)
        
        # Faint grey line extends infinitely through it
        line_start = origin + (origin - v_target) * 2.5
        line_end = origin + (v_target - origin) * 3.5
        inf_line = Line(line_start, line_end, color=LINE_COLOR, stroke_width=2).set_stroke(opacity=0.6)
        
        self.play(Create(v_vec))
        self.play(Create(inf_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Two non-parallel vectors can \"paint\" an entire 2D plane."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(W_COLOR)
        )
        
        w_target = self.grid["A3"]
        w_vec = Arrow(origin, w_target, buff=0, color=W_COLOR)
        
        # "Painting" square - starts small at origin and grows
        painting_square = Square(side_length=0.1, color=PLANE_COLOR, fill_opacity=0.3, stroke_width=0)
        painting_square.move_to(origin)
        
        self.play(Create(w_vec))
        # Grow to cover a significant part of the grid area
        self.play(
            painting_square.animate.stretch_to_fit_width(5.0).stretch_to_fit_height(4.5).move_to(self.grid["C4"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Think of Span as the full territory covered."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(PLANE_COLOR)
        )
        
        # Full plane fill represented by a rectangle
        full_plane = Rectangle(width=6, height=5, color=PLANE_COLOR, fill_opacity=0.5, stroke_width=0)
        # Resolved Issue 24: adjusted area and scale
        self.place_in_area(full_plane, 'B2', 'E5', scale_factor=0.85)
        
        self.play(FadeTransform(painting_square, full_plane))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "If vectors align, the Span collapses to a line."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLLAPSE_COLOR)
        )
        
        # Collapse w onto v's direction and the plane into a line
        # w is pointing at A3 (up), v is pointing at C5 (right). 
        # C3 to A3 is up. C3 to C5 is right. Rotation is -PI/2.
        self.play(
            Rotate(w_vec, angle=-PI/2, about_point=origin),
            full_plane.animate.stretch_to_fit_height(0.05).set_opacity(0.8).move_to(inf_line.get_center()),
            run_time=2.5
        )
        self.wait(2)
