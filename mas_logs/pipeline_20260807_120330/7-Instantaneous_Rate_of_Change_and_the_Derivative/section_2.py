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
        # Data from storyboard
        title = "Prerequisite Check: The Slope of a Line"
        lines = [
            "For a straight line, the slope is always constant.",
            "We calculate it using the simple rise over run.",
            "This value represents the constant average rate of change."
        ]
        
        # Colors based on animation description
        COLOR_RISE = "#00FF00"
        COLOR_RUN = "#0000FF"
        COLOR_FORMULA = "#FFFF00"
        COLOR_LINE = WHITE
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Description: A straight diagonal line appears on a Position-Time graph.
        
        # Define Axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GRAY},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT},
        )
        x_label = Text("Time (t)", font_size=16).next_to(axes.x_axis, DOWN)
        y_label = Text("Pos (s)", font_size=16).next_to(axes.y_axis, LEFT).rotate(90 * DEGREES)
        graph_group = VGroup(axes, x_label, y_label)
        
        # Issue 32/42: Place graph in area C3 to F6 with scale_factor=0.9
        # B021: Start visuals at Column 3 to protect lecture text margins.
        self.place_in_area(graph_group, "C3", "F6", scale_factor=0.9)
        
        # Define the diagonal line: s = t
        func_line = axes.plot(lambda t: t, x_range=[0, 4], color=COLOR_LINE)
        
        self.play(self.lecture[0].animate.set_color(COLOR_LINE))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(func_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Vertical 'Rise' in green (#00FF00) and horizontal 'Run' in blue (#0000FF) appear.
        
        # Define points for rise/run: from (1,1) to (3,3)
        p1 = axes.c2p(1, 1)
        p2 = axes.c2p(3, 3)
        p_corner = axes.c2p(3, 1)
        
        run_line = Line(p1, p_corner, color=COLOR_RUN, stroke_width=6)
        rise_line = Line(p_corner, p2, color=COLOR_RISE, stroke_width=6)
        
        run_label = Text("Run", font_size=18, color=COLOR_RUN).next_to(run_line, DOWN, buff=0.1)
        rise_label = Text("Rise", font_size=18, color=COLOR_RISE).next_to(rise_line, RIGHT, buff=0.1)
        
        self.play(self.lecture[1].animate.set_color(COLOR_RUN)) # Highlight text part 1
        self.play(Create(run_line), Write(run_label))
        self.play(self.lecture[1].animate.set_color(COLOR_RISE)) # Highlight text part 2
        self.play(Create(rise_line), Write(rise_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: The formula 'Slope = Rise / Run' appears in yellow (#FFFF00).
        
        # Issue 31/42: Place formula in area B4 to B6 with scale_factor=1.0
        # This keeps it away from the title (Row A) and aligns with the lecture notes.
        slope_formula = MathTex(
            r"\text{Slope} = \frac{\text{Rise}}{\text{Run}}", 
            color=COLOR_FORMULA
        )
        self.place_in_area(slope_formula, "B4", "B6", scale_factor=1.0)
        
        self.play(self.lecture[2].animate.set_color(COLOR_FORMULA))
        self.play(Write(slope_formula))
        self.wait(2)
