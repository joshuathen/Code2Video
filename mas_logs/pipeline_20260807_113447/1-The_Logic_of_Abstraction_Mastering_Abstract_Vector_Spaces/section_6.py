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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        # Topic: The Logic of Abstraction: Mastering Abstract Vector Spaces
        self.setup_layout(
            "The Digital Application: RGB Color Space",
            [
                "Screens treat colors as abstract vectors.",
                "Digital sliders perform scalar multiplication.",
                "Mixing colors is basic vector addition."
            ]
        )
        
        # Color palette
        r_color = "#FF4444" # Light Red
        g_color = "#44FF44" # Light Green
        b_color = "#4444FF" # Light Blue
        y_color = "#FFFF44" # Light Yellow
        
        # === Animation for Lecture Line 1 ===
        # screens treat colors as abstract vectors.
        # Three sliders labeled R, G, B appear.
        self.lecture[0].set_color(YELLOW)
        
        # Slider 1: Red
        r_line = Line(LEFT*0.6, RIGHT*0.6, color=r_color)
        r_knob = Dot(color=r_color)
        r_label = MathTex("R", color=r_color).scale(0.8)
        r_slider_group = VGroup(r_line, r_knob)
        
        # Slider 2: Green
        g_line = Line(LEFT*0.6, RIGHT*0.6, color=g_color)
        g_knob = Dot(color=g_color)
        g_label = MathTex("G", color=g_color).scale(0.8)
        g_slider_group = VGroup(g_line, g_knob)
        
        # Slider 3: Blue
        b_line = Line(LEFT*0.6, RIGHT*0.6, color=b_color)
        b_knob = Dot(color=b_color)
        b_label = MathTex("B", color=b_color).scale(0.8)
        b_slider_group = VGroup(b_line, b_knob)
        
        # Grid Placement - Columns 1 and 2 used for sliders and labels
        self.place_at_grid(r_slider_group, "B2", scale_factor=0.8)
        self.place_at_grid(r_label, "B1", scale_factor=0.8)
        self.place_at_grid(g_slider_group, "C2", scale_factor=0.8)
        self.place_at_grid(g_label, "C1", scale_factor=0.8)
        self.place_at_grid(b_slider_group, "D2", scale_factor=0.8)
        self.place_at_grid(b_label, "D1", scale_factor=0.8)
        
        # Initialize knob positions at the left end of lines
        r_knob.move_to(r_line.get_left())
        g_knob.move_to(g_line.get_left())
        b_knob.move_to(b_line.get_left())
        
        self.play(
            Create(r_slider_group), Write(r_label),
            Create(g_slider_group), Write(g_label),
            Create(b_slider_group), Write(b_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Digital sliders perform scalar multiplication.
        # Increasing R and G sliders creates a yellow square.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Color output display in the upper right
        color_square = Square(side_length=1.2, fill_opacity=1, stroke_width=2, color=BLACK)
        self.place_in_area(color_square, "B4", "C5")
        square_label = Text("Color Output", font_size=16).next_to(color_square, UP, buff=0.2)
        
        # Trackers for normalized R and G values (0 to 1)
        r_val = ValueTracker(0.0)
        g_val = ValueTracker(0.0)
        
        # Updaters for reactive visualization (avoiding always_redraw)
        r_knob.add_updater(lambda m: m.move_to(r_line.point_from_proportion(r_val.get_value())))
        g_knob.add_updater(lambda m: m.move_to(g_line.point_from_proportion(g_val.get_value())))
        
        def update_square(m):
            # Using rgb_to_color for smooth interpolation
            m.set_color(rgb_to_color([r_val.get_value(), g_val.get_value(), 0]))
            
        color_square.add_updater(update_square)
        
        self.play(FadeIn(color_square), FadeIn(square_label))
        # Animate "scalar multiplication" (increasing the R and G components)
        self.play(r_val.animate.set_value(1.0), run_time=1.5, rate_func=linear)
        self.play(g_val.animate.set_value(1.0), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Mixing colors is basic vector addition.
        # R-G-B coordinate triad shows vector pointing to yellow.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Axes at bottom right to avoid occlusion
        origin_pos = self.grid["F4"]
        
        r_axis_end = self.grid["F6"]
        g_axis_end = self.grid["D4"]
        b_axis_end = self.grid["E3"]
        
        r_axis = Arrow(origin_pos, r_axis_end, color=r_color, buff=0, stroke_width=4)
        g_axis = Arrow(origin_pos, g_axis_end, color=g_color, buff=0, stroke_width=4)
        b_axis = Arrow(origin_pos, b_axis_end, color=b_color, buff=0, stroke_width=4)
        
        axes_group = VGroup(r_axis, g_axis, b_axis)
        axes_labels = VGroup(
            MathTex("R", color=r_color).scale(0.6).next_to(r_axis_end, RIGHT, buff=0.1),
            MathTex("G", color=g_color).scale(0.6).next_to(g_axis_end, UP, buff=0.1),
            MathTex("B", color=b_color).scale(0.6).next_to(b_axis_end, UL, buff=0.1)
        )
        
        # Resultant yellow vector: addition of R and G components
        yellow_vec_end = origin_pos + (r_axis_end - origin_pos) + (g_axis_end - origin_pos)
        yellow_vec = Arrow(origin_pos, yellow_vec_end, color=y_color, buff=0, stroke_width=6)
        yellow_label = MathTex(r"\vec{R} + \vec{G}", color=y_color).scale(0.7).next_to(yellow_vec_end, UR, buff=0.1)
        
        # Component lines to show addition (parallelogram rule)
        comp_r = DashedLine(g_axis_end, yellow_vec_end, color=r_color, stroke_width=2)
        comp_g = DashedLine(r_axis_end, yellow_vec_end, color=g_color, stroke_width=2)
        
        self.play(Create(axes_group), Write(axes_labels))
        self.play(
            Create(comp_r), Create(comp_g),
            Create(yellow_vec), Write(yellow_label),
            run_time=2
        )
        self.wait(3)
