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
        title = "Mapping the Circle: Euler's General Formula"
        lecture_lines = [
            "Euler’s formula maps this circular path on a plane.",
            "The expression e to the ix tracks our position.",
            "Cosine measures how far we are horizontally.",
            "Sine measures our vertical height on the complex plane.",
            "Together, they define every point on the unit circle."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COS_COLOR = "#88CA5E"
        SIN_COLOR = "#58C4DD"
        RADIUS_COLOR = WHITE
        HIGHLIGHT_COLOR = YELLOW

        # Central Visual Setup
        # Origin is between C3 and D4
        origin_pos = (self.grid["C3"] + self.grid["D4"]) / 2
        
        # Axes and Plane (Issue 43: No MathTex)
        axes = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=3.0,
            y_length=3.0,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.place_in_area(axes, "B2", "E5")
        
        re_label = Text("Re", font_size=20)
        self.place_at_grid(re_label, "D6", scale_factor=0.8)
        
        im_label = Text("Im", font_size=20)
        # Issue 33: Fix: self.place_at_grid(im_label, 'A4', scale_factor=0.8)
        self.place_at_grid(im_label, "A4", scale_factor=0.8)
        
        # 1 unit in axes = 3.0 (length) / 2.4 (range) = 1.25
        radius_val = 1.25
        unit_circle = Circle(radius=radius_val, color=GREY_B, stroke_opacity=0.5)
        self.place_in_area(unit_circle, "B2", "E5")

        # State management
        angle_tracker = ValueTracker(PI/4)

        # Mobjects with updaters
        # NOTE: Using origin_pos derived from grid to maintain alignment
        
        radius_line = Line(origin_pos, origin_pos, color=RADIUS_COLOR)
        radius_line.add_updater(lambda l: l.set_points_as_corners([
            origin_pos, 
            origin_pos + np.array([
                np.cos(angle_tracker.get_value()) * radius_val, 
                np.sin(angle_tracker.get_value()) * radius_val, 
                0
            ])
        ]))

        dot = Dot(color=WHITE)
        dot.add_updater(lambda d: d.move_to(radius_line.get_end()))

        # Arc for the angle
        angle_arc = always_redraw(lambda: Arc(
            radius=0.4, 
            start_angle=0, 
            angle=angle_tracker.get_value(), 
            arc_center=origin_pos, 
            color=HIGHLIGHT_COLOR
        ))

        # Dynamic 'x' label for the angle
        x_label = Text("x", font_size=20, color=HIGHLIGHT_COLOR)
        x_label.add_updater(lambda m: m.move_to(origin_pos + 0.6 * np.array([
            np.cos(angle_tracker.get_value()/2), 
            np.sin(angle_tracker.get_value()/2), 
            0
        ])))

        # Label for the point on the circle
        e_label = MarkupText("e<sup>ix</sup>", font_size=24)
        e_label.add_updater(lambda m: m.move_to(dot.get_center() + 0.45 * (dot.get_center() - origin_pos)))

        # Cosine segment on real axis
        cos_line = Line(origin_pos, origin_pos, color=COS_COLOR, stroke_width=6)
        cos_line.add_updater(lambda l: l.set_points_as_corners([
            origin_pos, 
            origin_pos + np.array([np.cos(angle_tracker.get_value()) * radius_val, 0, 0])
        ]))

        # Issue 35: Fix: self.place_at_grid(cos_label, 'E5', scale_factor=0.6)
        cos_label = MarkupText("cos(x)", color=COS_COLOR, font_size=24)
        self.place_at_grid(cos_label, "E5", scale_factor=0.6)

        # Sine segment on imaginary axis
        sin_line = Line(origin_pos, origin_pos, color=SIN_COLOR, stroke_width=6)
        sin_line.add_updater(lambda l: l.set_points_as_corners([
            origin_pos, 
            origin_pos + np.array([0, np.sin(angle_tracker.get_value()) * radius_val, 0])
        ]))

        # Issue 34: Fix: self.place_at_grid(i_sin_label, 'C5', scale_factor=0.6)
        i_sin_label = MarkupText("i sin(x)", color=SIN_COLOR, font_size=24)
        self.place_at_grid(i_sin_label, "C5", scale_factor=0.6)

        # The main equation (Issue 43: No MathTex, use MarkupText)
        formula = MarkupText("e<sup>ix</sup> = cos(x) + i sin(x)", font_size=32)
        self.place_in_area(formula, "F2", "F5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "Euler’s formula maps this circular path on a plane."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(axes), Write(re_label), Write(im_label))
        self.play(Create(unit_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The expression e to the ix tracks our position."
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(radius_line), Create(dot), Create(angle_arc), Write(x_label))
        self.play(Write(e_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Cosine measures how far we are horizontally."
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(COS_COLOR))
        self.play(Create(cos_line), Write(cos_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Sine measures our vertical height on the complex plane."
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(SIN_COLOR))
        self.play(Create(sin_line), Write(i_sin_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Together, they define every point on the unit circle."
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Write(formula))
        
        # Rotation animation to demonstrate the relationship
        self.play(angle_tracker.animate.set_value(5*PI/6), run_time=3, rate_func=smooth)
        self.play(angle_tracker.animate.set_value(PI/6), run_time=2, rate_func=smooth)
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
