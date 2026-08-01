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
        # Initial layout setup
        title_text = "The Formal Theorem: Scaling Up Dimensions"
        lecture_lines = [
            "Borsuk-Ulam maps an n-sphere to n-dimensional space.",
            "On Earth, two opposites share temperature and pressure.",
            "Every continuous mapping has a matching antipodal pair."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Text: 'For f: S^n -> R^n, exists x s.t. f(x) = f(-x)'
        # Resolved Issue 32: Positioning and scaling for formula
        formula = Text("f: S^n -> R^n, exists x s.t. f(x) = f(-x)", color="#FFFFFF", font_size=25)
        self.place_in_area(formula, "A3", "B6", scale_factor=0.9)
        
        self.play(
            Write(formula),
            self.lecture[0].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A 2D sphere (Earth) shows two values: Temperature and Pressure
        # Resolved Issue 31: Moving Earth to D2-F5 to avoid overlap with C-row labels
        earth = Circle(radius=1.3, color=WHITE, stroke_width=2)
        self.place_in_area(earth, "D2", "F5", scale_factor=1.0)
        earth_center = earth.get_center()

        temp_label = Text("Temperature", font_size=18, color="#FF4500")
        pres_label = Text("Pressure", font_size=18, color="#1E90FF")
        
        self.place_at_grid(temp_label, "C2", scale_factor=1.0)
        self.place_at_grid(pres_label, "C5", scale_factor=1.0)

        # Decorative data points on the sphere
        points_data = VGroup()
        for angle in np.linspace(0, 2*PI, 8, endpoint=False):
            p = earth_center + np.array([np.cos(angle)*1.3, np.sin(angle)*1.3, 0])
            dot = Dot(p, radius=0.03, color=GREY_A)
            points_data.add(dot)

        self.play(
            Create(earth),
            FadeIn(temp_label),
            FadeIn(pres_label),
            FadeIn(points_data),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A scanning crosshair finds a pair of antipodal points
        
        def create_crosshair(point, color=WHITE):
            h = Line(point + LEFT*0.2, point + RIGHT*0.2, color=color, stroke_width=2)
            v = Line(point + UP*0.2, point + DOWN*0.2, color=color, stroke_width=2)
            return VGroup(h, v)

        # Start scanning from vertical position
        ch1 = create_crosshair(earth_center + UP * 1.3)
        ch2 = create_crosshair(earth_center + DOWN * 1.3)
        scanner = VGroup(ch1, ch2)
        
        self.play(
            Create(scanner),
            self.lecture[2].animate.set_color("#FFFFFF")
        )

        # Scan around the circle by 135 degrees
        self.play(
            Rotate(scanner, angle=-135*DEGREES, about_point=earth_center),
            run_time=2.5,
            rate_func=bezier([0, 0, 1, 1])
        )

        # Resolved Issue 33: Scaled match text at F3
        match_highlight1 = create_crosshair(ch1.get_center(), color=YELLOW).scale(1.2)
        match_highlight2 = create_crosshair(ch2.get_center(), color=YELLOW).scale(1.2)
        
        match_text = Text("VALUES MATCH!", font_size=22, color=YELLOW)
        self.place_at_grid(match_text, "F3", scale_factor=0.7)

        # Values display at match point
        val_display = VGroup(
            Text("T1 = T2", font_size=16, color="#FF4500"),
            Text("P1 = P2", font_size=16, color="#1E90FF")
        ).arrange(DOWN, buff=0.1)
        # Positioned via grid to avoid manual placement constraints
        self.place_at_grid(val_display, "F6", scale_factor=1.0)

        self.play(
            Transform(ch1, match_highlight1),
            Transform(ch2, match_highlight2),
            FadeIn(match_text),
            FadeIn(val_display)
        )
        self.wait(2)
