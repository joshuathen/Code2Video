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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "The Change of Basis Matrix (The Bridge)"
        lines = [
            "The change of basis matrix acts as a translator.",
            "Its columns are the new basis vectors in standard coordinates.",
            "Each column tells where a new basis unit lands."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_B1 = "#FF0000"
        COLOR_B2 = "#FF8C00"
        COLOR_HIGHLIGHT = YELLOW
        COLOR_STANDARD = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Create Coordinate System (Axes)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        axes_label = axes.get_axis_labels(x_label=Text("x"), y_label=Text("y")).scale(0.6)
        axes_group = VGroup(axes, axes_label)
        # Issue 36: Adjusted area to C3-F6 and scale 0.9
        self.place_in_area(axes_group, "C3", "F6", scale_factor=0.9)
        
        # Define vectors b1 and b2 in standard coordinates
        vec_b1 = Arrow(axes.c2p(0, 0), axes.c2p(2, 1), buff=0, color=COLOR_STANDARD)
        vec_b2 = Arrow(axes.c2p(0, 0), axes.c2p(-1, 1), buff=0, color=COLOR_STANDARD)
        
        label_b1 = Text("b1", color=COLOR_STANDARD, font_size=24)
        label_b2 = Text("b2", color=COLOR_STANDARD, font_size=24)
        
        label_b1.next_to(vec_b1.get_end(), RIGHT, buff=0.1)
        label_b2.next_to(vec_b2.get_end(), LEFT, buff=0.1)
        
        # Bridge asset integration (Issue 26)
        bridge_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg")
        bridge_asset.set_color(GREY_B)
        self.place_at_grid(bridge_asset, "A6", scale_factor=0.5)

        self.play(Create(axes_group))
        self.play(
            GrowArrow(vec_b1),
            Write(label_b1),
            GrowArrow(vec_b2),
            Write(label_b2),
            FadeIn(bridge_asset),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(COLOR_STANDARD)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Construct Matrix P
        p_matrix_label = Text("P =", font_size=32)
        
        # Elements (Columns of P are b1=[2,1] and b2=[-1,1])
        t2 = Text("2", font_size=28)
        t1_bot = Text("1", font_size=28)
        t_minus_1 = Text("-1", font_size=28)
        t1_top = Text("1", font_size=28)
        
        col1_vals = VGroup(t2, t1_bot).arrange(DOWN, buff=0.4)
        col2_vals = VGroup(t_minus_1, t1_top).arrange(DOWN, buff=0.4)
        matrix_elements = VGroup(col1_vals, col2_vals).arrange(RIGHT, buff=0.6)
        
        # Manual brackets
        l_bracket = Text("[", font_size=48).scale([0.8, 1.8, 1]).next_to(matrix_elements, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=48).scale([0.8, 1.8, 1]).next_to(matrix_elements, RIGHT, buff=0.1)
        
        p_matrix = VGroup(l_bracket, matrix_elements, r_bracket)
        matrix_vgroup = VGroup(p_matrix_label, p_matrix).arrange(RIGHT, buff=0.2)
        
        # Issue 35: Adjusted area to A2-B5 and scale 0.8
        self.place_in_area(matrix_vgroup, "A2", "B5", scale_factor=0.8)
        
        self.play(Write(p_matrix_label), FadeIn(l_bracket), FadeIn(r_bracket))
        self.play(Write(col1_vals), Write(col2_vals), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(COLOR_STANDARD)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Highlight logic: column 1 matches b1, column 2 matches b2
        rect1 = SurroundingRectangle(col1_vals, color=COLOR_B1, buff=0.1)
        rect2 = SurroundingRectangle(col2_vals, color=COLOR_B2, buff=0.1)
        
        self.play(
            vec_b1.animate.set_color(COLOR_B1),
            label_b1.animate.set_color(COLOR_B1),
            col1_vals.animate.set_color(COLOR_B1),
            Create(rect1),
            run_time=1
        )
        self.play(
            vec_b2.animate.set_color(COLOR_B2),
            label_b2.animate.set_color(COLOR_B2),
            col2_vals.animate.set_color(COLOR_B2),
            Create(rect2),
            run_time=1
        )
        
        self.wait(2)
