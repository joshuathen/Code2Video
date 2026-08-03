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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "Span: The Reachable Map"
        lecture_lines = [
            "Span is the set of all possible vector combinations.",
            "One vector alone spans only a single line.",
            "Two non-parallel vectors can span an entire plane.",
            "The span shows every location our robot can reach."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define Colors
        COLOR_V = "#00FF00"          # Green for vector v
        COLOR_W = "#FFFF00"          # Yellow for vector w
        COLOR_SPAN_LINE = "#AAAAAA"  # Gray for line span
        COLOR_SPAN_PLANE = "#444444" # Dark gray for plane span
        COLOR_HIGHLIGHT = "#00FFFF"  # Cyan for final span highlight

        # === Animation for Lecture Line 1 ===
        # "Span is the set of all possible vector combinations."
        # Initialize the coordinate system and generic formula.
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Fix Issue 26: Scale axes to 0.75
        self.place_in_area(axes, "B2", "E5", scale_factor=0.75)
        
        span_formula = MathTex(
            r"Span(\mathbf{v}, \mathbf{w}) = \{c_1\mathbf{v} + c_2\mathbf{w}\}", 
            font_size=32
        )
        # Fix Issue 25: Reposition span_formula
        self.place_in_area(span_formula, "A2", "A5", scale_factor=0.7)
        
        # Asset: Robot Icon
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.scale(0.2)
        robot.move_to(axes.c2p(0, 0))

        self.play(
            Create(axes),
            Write(span_formula),
            FadeIn(robot),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "One vector alone spans only a single line."
        # Show vector v (1,0) in #00FF00 and a horizontal line in #AAAAAA.
        self.play(self.lecture[1].animate.set_color(COLOR_V))
        
        v_vec = Arrow(axes.c2p(0, 0), axes.c2p(1, 0), buff=0, color=COLOR_V)
        v_label = MathTex(r"\mathbf{v}", color=COLOR_V, font_size=24).next_to(v_vec, UP, buff=0.1)
        
        # Horizontal line representing the span of v
        span_line = Line(axes.c2p(-3, 0), axes.c2p(3, 0), color=COLOR_SPAN_LINE, stroke_width=2)
        
        self.play(
            GrowArrow(v_vec),
            Write(v_label),
            robot.animate.move_to(axes.c2p(1, 0)) # Move robot to tip of v
        )
        self.play(Create(span_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Two non-parallel vectors can span an entire plane."
        # Add vector w (0,1) and shade the entire 2D plane in #444444.
        self.play(self.lecture[2].animate.set_color(COLOR_W))
        
        w_vec = Arrow(axes.c2p(0, 0), axes.c2p(0, 1), buff=0, color=COLOR_W)
        w_label = MathTex(r"\mathbf{w}", color=COLOR_W, font_size=24).next_to(w_vec, LEFT, buff=0.1)
        
        # Shaded plane background (sized to match axes area)
        # Area from B2 to E5 center is approximately 3.0 units wide/high in the grid
        plane_shade = Rectangle(
            width=3.0,
            height=3.0,
            fill_color=COLOR_SPAN_PLANE,
            fill_opacity=0.6,
            stroke_width=0
        ).move_to(axes.get_center())
        
        self.play(
            GrowArrow(w_vec),
            Write(w_label)
        )
        self.play(FadeIn(plane_shade))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The span shows every location our robot can reach."
        # Use Cyan to highlight the reachable area.
        self.play(
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT),
            plane_shade.animate.set_fill(color=COLOR_HIGHLIGHT, opacity=0.3)
        )
        # Move robot around the span to show reachability
        self.play(
            robot.animate.move_to(axes.c2p(-1.5, 1.5)),
            run_time=1
        )
        self.play(
            robot.animate.move_to(axes.c2p(1.5, -1.0)),
            run_time=1
        )
        self.play(Indicate(plane_shade, color=COLOR_HIGHLIGHT))
        self.wait(2)
