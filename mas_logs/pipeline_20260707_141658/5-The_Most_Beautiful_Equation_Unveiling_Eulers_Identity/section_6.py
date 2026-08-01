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
        # Data from shared state
        title_text = "The Destination: When x = Pi"
        lecture_lines = [
            "Set the rotation angle to exactly pi radians.",
            "This is a half-turn around the unit circle.",
            "We land exactly at negative one on the axis."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors
        color_line1 = "#FFFFFF"  # White
        color_line2 = "#FFFFE0"  # Light Yellow
        color_line3 = "#ADD8E6"  # Light Blue

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(color_line1))
        
        # Setup unit circle and axes in the grid area
        plane = ComplexPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, "A1", "F6", scale_factor=0.8)
        
        circle = Circle(radius=plane.get_x_unit_size(), color=WHITE, stroke_opacity=0.5)
        circle.move_to(plane.n2p(0))
        
        dot = Dot(point=plane.n2p(1), color=color_line1)
        # Using Text for robust rendering across different environments
        dot_label = Text("1", font_size=24, color=color_line1).next_to(dot, RIGHT, buff=0.1)
        
        self.play(Create(plane), Create(circle))
        self.play(FadeIn(dot), FadeIn(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(color_line2))
        
        # Animate point traveling half the circumference
        angle_tracker = ValueTracker(0)
        
        # Define the point position based on angle
        def get_point_on_circle():
            angle = angle_tracker.get_value()
            return plane.n2p(np.exp(1j * angle))
        
        # Using add_updater for the dot to follow the tracker
        dot.add_updater(lambda d: d.move_to(get_point_on_circle()))
        
        # Arc to trace the path - simple geometry in always_redraw is acceptable
        arc = always_redraw(lambda: Arc(
            radius=plane.get_x_unit_size(),
            start_angle=0,
            angle=angle_tracker.get_value(),
            color=color_line2,
            stroke_width=4
        ).move_to(plane.n2p(0)))
        
        self.add(arc)
        self.play(
            angle_tracker.animate.set_value(PI),
            dot.animate.set_color(color_line2),
            run_time=3, 
            rate_func=linear
        )
        dot.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(color_line3))
        
        # Point stops at -1, equation fades in
        dot.set_color(color_line3)
        final_label = Text("-1", font_size=24, color=color_line3).next_to(dot, LEFT, buff=0.1)
        
        # Fixed positioning for the equation to avoid overlap with unit circle (Issue #41)
        equation = Text("e^iπ = -1", font_size=36, color=color_line3)
        self.place_at_grid(equation, "A2", scale_factor=0.9)
        
        self.play(FadeIn(final_label))
        self.play(FadeIn(equation))
        self.wait(2)
