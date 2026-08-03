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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title_text = "Prerequisite: Discrete Signals and Indices"
        lecture_lines = [
            "Discrete signals are represented as arrays of numbers.",
            "Input signal x[n] meets impulse response h[n].",
            "The index n represents discrete steps in time."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_X = "#FFFFFF" # White
        COLOR_H = "#90EE90" # Light Green
        COLOR_N = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        # Draw a discrete sequence x[n] as white bars (#FFFFFF) with labels on the x-axis.
        self.play(self.lecture[0].animate.set_color(COLOR_X))
        
        x_vals = [1, 2, 3]
        x_axes = Line(LEFT*2, RIGHT*2, color=WHITE)
        x_bars = VGroup()
        x_labels = VGroup()
        
        for i, val in enumerate(x_vals):
            # Normalizing height slightly for visibility
            bar = Line(start=ORIGIN, end=UP*val*0.6, color=COLOR_X, stroke_width=10)
            # Use point_from_proportion for spacing
            bar.move_to(x_axes.point_from_proportion(i/2), aligned_edge=DOWN)
            lbl = MathTex(f"x[{i}]", font_size=24, color=COLOR_X).next_to(bar, DOWN, buff=0.1)
            x_bars.add(bar)
            x_labels.add(lbl)
            
        x_plot = VGroup(x_axes, x_bars, x_labels)
        # Resolved Issue 21: x_plot to 'A3'-'C6' with scale 0.7
        self.place_in_area(x_plot, 'A3', 'C6', scale_factor=0.7)
        
        self.play(Create(x_axes), Create(x_bars), Write(x_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a second sequence h[n] in light green (#90EE90) labeled 'Impulse Response'.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_H)
        )
        
        h_vals = [0.5, 1.0]
        h_axes = Line(LEFT*2, RIGHT*2, color=WHITE)
        h_bars = VGroup()
        h_labels = VGroup()
        
        for i, val in enumerate(h_vals):
            bar = Line(start=ORIGIN, end=UP*val*0.6, color=COLOR_H, stroke_width=10)
            # Two points: i=0 -> 0.0, i=1 -> 1.0
            bar.move_to(h_axes.point_from_proportion(i/1), aligned_edge=DOWN)
            lbl = MathTex(f"h[{i}]", font_size=24, color=COLOR_H).next_to(bar, DOWN, buff=0.1)
            h_bars.add(bar)
            h_labels.add(lbl)
            
        h_title = Text("Impulse Response", font_size=28, color=COLOR_H)
        h_plot = VGroup(h_axes, h_bars, h_labels, h_title)
        h_title.next_to(h_axes, UP, buff=0.5)
        
        # Resolved Issue 22: h_plot to 'E3'-'F6' with scale 0.7
        self.place_in_area(h_plot, 'E3', 'F6', scale_factor=0.7)
        
        self.play(Create(h_axes), Create(h_bars), Write(h_labels), Write(h_title))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the index n on the x-axis with a pulsing yellow dot (#FFFF00).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_N)
        )
        
        # Position dot on x_axes at index 1 (middle point)
        dot_pos = x_axes.point_from_proportion(0.5)
        dot = Dot(point=dot_pos, color=COLOR_N, radius=0.1)
        n_lbl = MathTex("n", color=COLOR_N, font_size=32).next_to(dot, UP, buff=0.2)
        
        self.play(FadeIn(dot), Write(n_lbl))
        
        # Pulsing effect using ValueTracker for compliance
        pulse_tracker = ValueTracker(1)
        dot.add_updater(lambda d: d.set_width(0.2 * pulse_tracker.get_value()))
        
        # Pulse twice
        self.play(pulse_tracker.animate.set_value(2), run_time=0.6, rate_func=there_and_back)
        self.play(pulse_tracker.animate.set_value(2), run_time=0.6, rate_func=there_and_back)
        
        dot.clear_updaters()
        self.wait(2)
        
        # Reset color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
