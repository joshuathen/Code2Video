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
        # Initializing the layout with the title and lecture lines
        title_text = "The Borsuk-Ulam Theorem: Antipodal Points"
        lecture_lines = [
            "Imagine two opposite points on the Earth's surface.",
            "These are called antipodal points on a sphere.",
            "Borsuk-Ulam says two opposite points share identical weather.",
            "Temperature and pressure match at these two locations.",
            "A continuous mapping always finds this balanced pair."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display a 2D blue circle representing Earth (#0000FF) with weather labels 'T' and 'P' scattered on it.
        self.play(self.lecture[0].animate.set_color("#0000FF"))
        earth = Circle(radius=1.5, color="#0000FF", fill_opacity=0.1)
        self.place_in_area(earth, 'B2', 'E5')
        
        weather_labels = VGroup()
        # Scattering background 'T' and 'P' labels inside the circular area
        pos_keys = ["B3", "B4", "C3", "C4", "E3", "E4"]
        for pk in pos_keys:
            label = Text("T, P", font_size=14, color=WHITE, fill_opacity=0.5)
            self.place_at_grid(label, pk)
            weather_labels.add(label)

        self.play(Create(earth), FadeIn(weather_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a diameter line through the center, highlighting the two endpoints in #FF0000.
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        point_left = Dot(color="#FF0000", radius=0.15)
        point_right = Dot(color="#FF0000", radius=0.15)
        
        # Position points at opposite sides of the circle on the grid
        self.place_at_grid(point_left, 'D2')
        self.place_at_grid(point_right, 'D5')
        
        diameter = Line(point_left.get_center(), point_right.get_center(), color=WHITE)
        
        self.play(Create(diameter))
        self.play(FadeIn(point_left), FadeIn(point_right))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Borsuk-Ulam says two opposite points share identical weather.
        # Keeping this line white for contrast against specific measurements.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Weather value displays: Temperature (T) and Pressure (P)
        # Explicitly setting mob_class=Text to avoid LaTeX dependency error
        t_left = DecimalNumber(14.2, num_decimal_places=1, color=YELLOW, font_size=24, mob_class=Text)
        p_left = DecimalNumber(1013.2, num_decimal_places=1, color=BLUE_B, font_size=24, mob_class=Text)
        t_right = DecimalNumber(22.8, num_decimal_places=1, color=YELLOW, font_size=24, mob_class=Text)
        p_right = DecimalNumber(998.4, num_decimal_places=1, color=BLUE_B, font_size=24, mob_class=Text)
        
        v_left = VGroup(
            VGroup(Text("T:", font_size=16), t_left).arrange(RIGHT, buff=0.1),
            VGroup(Text("P:", font_size=16), p_left).arrange(RIGHT, buff=0.1)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        v_right = VGroup(
            VGroup(Text("T:", font_size=16), t_right).arrange(RIGHT, buff=0.1),
            VGroup(Text("P:", font_size=16), p_right).arrange(RIGHT, buff=0.1)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        # Place reading values next to the antipodal points
        self.place_at_grid(v_left, 'D1')
        self.place_at_grid(v_right, 'D6')
        
        self.play(FadeIn(v_left), FadeIn(v_right))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate the 'T' and 'P' values at these endpoints changing until they are exactly equal.
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        self.play(
            t_left.animate.set_value(18.5),
            t_right.animate.set_value(18.5),
            p_left.animate.set_value(1005.8),
            p_right.animate.set_value(1005.8),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the labels 'Antipodal Balance' at the matching points in #00FF00.
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        label_l = Text("Antipodal\nBalance", font_size=14, color="#00FF00", line_spacing=0.8)
        label_r = Text("Antipodal\nBalance", font_size=14, color="#00FF00", line_spacing=0.8)
        
        # Position labels above the endpoints
        self.place_at_grid(label_l, 'C2')
        self.place_at_grid(label_r, 'C5')
        
        self.play(FadeIn(label_l), FadeIn(label_r))
        self.play(Indicate(label_l), Indicate(label_r))
        self.wait(2)
