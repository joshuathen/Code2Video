from manim import *
import numpy as np

# Fix: Removing curly braces from the input file path to prevent Manim's internal config 
# from attempting to format them as template placeholders in a loop.
if config.get("input_file"):
    config["input_file"] = str(config["input_file"]).replace("{", "").replace("}", "")

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
        # Setup context
        lecture_lines = [
            "Euler's formula defines movement along a circle.",
            "e^(iθ) equals cosine theta plus i sine theta.",
            "Real growth pushes out, but imaginary growth rotates.",
            "This traces a path around the origin perfectly.",
            "The result is an orbit at constant distance."
        ]
        self.setup_layout("The Geometric Magic: e^(iθ) as Rotation", lecture_lines)

        # Global ValueTracker for theta
        theta_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Formula at the top - Fixed Issue 28: A1 to A6, scale 0.9
        formula = Text("e^(iθ) = cos(θ) + i sin(θ)", font_size=30, color=WHITE)
        self.place_in_area(formula, "A1", "A6", scale_factor=0.9)
        
        # Unit circle background - Fixed Issue 29: B1 to F6, scale 0.85
        circle_bg = Circle(radius=1.8, color="#555555", stroke_width=2)
        self.place_in_area(circle_bg, "B1", "F6", scale_factor=0.85)
        plot_center = circle_bg.get_center()
        
        # Axes aligned with circle center
        axes = Axes(
            x_range=[-1.2, 1.2], y_range=[-1.2, 1.2],
            x_length=3.6, y_length=3.6,
            axis_config={"include_tip": False, "color": GREY_D}
        ).move_to(plot_center)
        
        self.play(Write(formula))
        self.play(Create(circle_bg), Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEAL)
        
        # Point at the start (1,0) relative to axes
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value()))))
        
        # One label - Positioned at x=1
        one_label = Text("1", font_size=20, color=GREY_A)
        one_label.next_to(axes.c2p(1,0), DOWN, buff=0.1)
        
        self.play(FadeIn(dot), Write(one_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Vector from origin
        vector = Line(plot_center, dot.get_center(), color=YELLOW, buff=0).add_tip(tip_length=0.15)
        vector.add_updater(lambda v: v.become(Line(plot_center, dot.get_center(), color=YELLOW, buff=0).add_tip(tip_length=0.15)))
        
        # Angle arc and label
        angle_arc = Arc(radius=0.4, start_angle=0, angle=0, arc_center=plot_center, color=WHITE)
        angle_arc.add_updater(lambda a: a.become(Arc(
            radius=0.4, 
            start_angle=0, 
            angle=theta_tracker.get_value(), 
            arc_center=plot_center, 
            color=WHITE
        )))
        
        theta_label = Text("θ", font_size=24, color=WHITE)
        theta_label.add_updater(lambda t: t.move_to(
            plot_center + 0.6 * np.array([
                np.cos(theta_tracker.get_value() / 2), 
                np.sin(theta_tracker.get_value() / 2), 
                0
            ])
        ))
        
        self.add(vector, angle_arc, theta_label)
        self.play(theta_tracker.animate.set_value(PI/3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE_C)
        
        # Projection lines (cos and sin)
        h_line = DashedLine(dash_length=0.05)
        v_line = DashedLine(dash_length=0.05)
        
        h_line.add_updater(lambda l: l.become(DashedLine(
            start=axes.c2p(np.cos(theta_tracker.get_value()), 0),
            end=axes.c2p(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value())),
            color=TEAL, stroke_width=2
        )))
        
        v_line.add_updater(lambda l: l.become(DashedLine(
            start=axes.c2p(0, np.sin(theta_tracker.get_value())),
            end=axes.c2p(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value())),
            color=BLUE, stroke_width=2
        )))
        
        # Labels for projection lines
        cos_label = Text("cos(θ)", font_size=16, color=TEAL)
        cos_label.add_updater(lambda c: c.next_to(axes.c2p(np.cos(theta_tracker.get_value())/2, 0), DOWN, buff=0.1))
        
        sin_label = Text("sin(θ)", font_size=16, color=BLUE)
        sin_label.add_updater(lambda s: s.next_to(axes.c2p(0, np.sin(theta_tracker.get_value())/2), LEFT, buff=0.1))

        self.play(Create(h_line), Create(v_line), Write(cos_label), Write(sin_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PINK)
        
        # Trace path
        trace = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=3)
        self.add(trace)
        
        # Complete the orbit (full circle from current PI/3 to 2*PI + PI/3)
        self.play(theta_tracker.animate.set_value(2*PI + PI/3), run_time=5, rate_func=linear)
        self.wait(2)
