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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "The Bridge: From Counting to Measuring"
        lines = [
            "Discrete variables count things like Bolt's battery packs.",
            "Continuous variables measure quantities like precise time.",
            "We measure 'how much' instead of 'how many'.",
            "Time and distance can take any value in a range.",
            "This requires a smooth curve instead of bars."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Create 5 blue (#0000FF) battery icons on the left side of the grid.
        self.lecture[0].set_color("#0000FF")
        
        batteries = VGroup()
        for i in range(1, 6): # Rows B to F, Column 1
            rect = Rectangle(width=0.4, height=0.6, color="#0000FF", fill_opacity=0.5)
            tip = Rectangle(width=0.2, height=0.1, color="#0000FF", fill_opacity=1).next_to(rect, UP, buff=0)
            battery = VGroup(rect, tip)
            self.place_at_grid(battery, f"{chr(65+i)}1", scale_factor=0.7)
            batteries.add(battery)
        
        self.play(Create(batteries))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a green (#00FF00) stopwatch on the right side with counting decimals.
        self.lecture[1].set_color("#00FF00")
        
        stopwatch_body = Circle(radius=0.5, color="#00FF00")
        stopwatch_top = Rectangle(width=0.15, height=0.1, color="#00FF00", fill_opacity=1).next_to(stopwatch_body, UP, buff=0)
        stopwatch_needle = Line(stopwatch_body.get_center(), stopwatch_body.get_top(), color="#00FF00")
        stopwatch_label = DecimalNumber(0, num_decimal_places=4, color="#00FF00", mob_class=Text).scale(0.6).next_to(stopwatch_body, DOWN)
        
        stopwatch = VGroup(stopwatch_body, stopwatch_top, stopwatch_needle, stopwatch_label)
        # Fix for Issue 21: Position stopwatch in the top right corner (A5-B6 area)
        self.place_in_area(stopwatch, 'A5', 'B6', scale_factor=0.8)
        
        time_tracker = ValueTracker(0)
        # Use updater for the decimal number and needle rotation
        stopwatch_label.add_updater(lambda d: d.set_value(time_tracker.get_value()))
        # Center of rotation must be dynamic
        stopwatch_needle.add_updater(lambda n: n.set_angle(PI/2 - time_tracker.get_value() * PI, about_point=stopwatch_body.get_center()))

        self.play(Create(stopwatch_body), Create(stopwatch_top), Create(stopwatch_needle), Write(stopwatch_label))
        self.play(time_tracker.animate.set_value(1.2345), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade out the battery icons while scaling up the stopwatch.
        self.lecture[2].set_color(WHITE)
        self.play(
            FadeOut(batteries),
            stopwatch.animate.scale(1.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The stopwatch decimals extend into a smooth line on the x-axis.
        self.lecture[3].set_color(WHITE)
        
        axis = Line(self.grid["F1"], self.grid["F6"], color=WHITE)
        
        # Stop updaters before movement
        stopwatch_label.clear_updaters()
        stopwatch_needle.clear_updaters()
        
        self.play(
            FadeOut(stopwatch_body, stopwatch_top, stopwatch_needle),
            stopwatch_label.animate.move_to(self.grid["F3"]).set_color(WHITE).scale(0.8),
            Create(axis)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A smooth curve fades in above the line to represent continuity.
        self.lecture[4].set_color(YELLOW)
        
        # Define curve using FunctionGraph with respect to scene coordinates
        # Grid F1 to F6 is x=0.5 to x=5.5 at y=-2.8
        curve = FunctionGraph(
            lambda x: 3.0 * np.exp(-((x - 3.0)**2) / 2.0) - 2.8,
            x_range=[0.5, 5.5],
            color=YELLOW
        )
        
        self.play(Create(curve))
        self.wait(2)
