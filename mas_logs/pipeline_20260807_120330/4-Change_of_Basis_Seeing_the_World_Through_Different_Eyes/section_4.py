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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines from the storyboard
        title = "The Translation Manual: The Change of Basis Matrix"
        lines = [
            "Matrix P bridges these two different coordinate worlds.",
            "Its columns are new basis vectors in standard terms.",
            "This matrix serves as a translation manual between grids."
        ]
        self.setup_layout(title, lines)
        
        # Colors defined in the storyboard
        COLOR_I = "#0000FF"   # Blue
        COLOR_J = "#00FF00"   # Green
        COLOR_B1 = "#FFA500"  # Orange
        COLOR_B2 = "#00FFFF"  # Cyan
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Create Axes for the visualization
        # Adjusted placement based on Issue 33 (B1-F3, scale 0.9)
        axes = Axes(
            x_range=[-2, 3, 1],
            y_range=[-1, 2, 1],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": True, "color": GREY_C},
        )
        self.place_in_area(axes, 'B1', 'F3', scale_factor=0.9)
        
        # Standard basis vectors i and j
        vec_i = Arrow(axes.c2p(0, 0), axes.c2p(1, 0), buff=0, color=COLOR_I)
        vec_j = Arrow(axes.c2p(0, 0), axes.c2p(0, 1), buff=0, color=COLOR_J)
        
        # Labels for standard basis
        lbl_i = MathTex("i", color=COLOR_I, font_size=24).next_to(vec_i.get_end(), DOWN, buff=0.1)
        lbl_j = MathTex("j", color=COLOR_J, font_size=24).next_to(vec_j.get_end(), LEFT, buff=0.1)
        
        # Pirate basis vectors b1 and b2
        vec_b1 = Arrow(axes.c2p(0, 0), axes.c2p(2, 1), buff=0, color=COLOR_B1)
        vec_b2 = Arrow(axes.c2p(0, 0), axes.c2p(-1, 1), buff=0, color=COLOR_B2)
        
        # Labels for Pirate basis
        lbl_b1 = MathTex("b_1", color=COLOR_B1, font_size=24).next_to(vec_b1.get_end(), RIGHT, buff=0.1)
        lbl_b2 = MathTex("b_2", color=COLOR_B2, font_size=24).next_to(vec_b2.get_end(), UP, buff=0.1)
        
        self.play(Create(axes))
        self.play(
            Create(vec_i), Create(vec_j), 
            Write(lbl_i), Write(lbl_j),
            run_time=1.5
        )
        self.play(
            Create(vec_b1), Create(vec_b2), 
            Write(lbl_b1), Write(lbl_b2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transitions lecture focus
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight standard coordinates of b1 [2, 1]
        dash_b1_x = DashedLine(axes.c2p(2, 0), axes.c2p(2, 1), color=COLOR_B1)
        dash_b1_y = DashedLine(axes.c2p(0, 1), axes.c2p(2, 1), color=COLOR_B1)
        
        # Highlight standard coordinates of b2 [-1, 1]
        dash_b2_x = DashedLine(axes.c2p(-1, 0), axes.c2p(-1, 1), color=COLOR_B2)
        dash_b2_y = DashedLine(axes.c2p(0, 1), axes.c2p(-1, 1), color=COLOR_B2)
        
        self.play(Create(dash_b1_x), Create(dash_b1_y))
        self.play(Create(dash_b2_x), Create(dash_b2_y))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transitions lecture focus
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Form matrix P = [[2, -1], [1, 1]]
        # The columns of P are b1=[2,1] and b2=[-1,1]
        matrix_p = Matrix(
            [[2, -1], [1, 1]],
            left_bracket="[",
            right_bracket="]",
            element_to_mobject_config={"font_size": 32}
        )
        
        # Color the columns of the matrix to match vectors
        matrix_p.get_columns()[0].set_color(COLOR_B1)
        matrix_p.get_columns()[1].set_color(COLOR_B2)
        
        # Matrix Label Group
        p_label = MathTex("P = ", font_size=32)
        p_group = VGroup(p_label, matrix_p).arrange(RIGHT, buff=0.1)
        
        # Position the p_group according to Issue 32 (C5-E6, scale 0.85)
        self.place_in_area(p_group, 'C5', 'E6', scale_factor=0.85)
        
        # Column indicators/labels for the matrix
        b1_col_lbl = MathTex("b_1", color=COLOR_B1, font_size=24)
        b2_col_lbl = MathTex("b_2", color=COLOR_B2, font_size=24)
        
        # Align labels above the columns
        b1_col_lbl.next_to(matrix_p.get_columns()[0], UP, buff=0.2)
        b2_col_lbl.next_to(matrix_p.get_columns()[1], UP, buff=0.2)
        
        self.play(Write(p_group))
        self.play(Write(b1_col_lbl), Write(b2_col_lbl))
        self.wait(2)
        
        # Reset lecture line color
        self.lecture[2].set_color(WHITE)
        self.wait(1)
