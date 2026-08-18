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
        # Title and Lecture Lines
        title = "The Integral: Accumulating the Area"
        lecture_lines = [
            "Now, let's observe Dash's velocity graph instead.",
            "The area under this curve represents his total displacement.",
            "Imagine a paint roller moving across the time axis.",
            "Each sliver of area adds to his total distance.",
            "Integration sums these infinite tiny slices of progress."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        LIME = "#7CFC00"
        YELLOW = "#FFFF00"
        ORANGE = "#FF8C00"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # A lime green (#7CFC00) velocity curve is drawn on the screen.
        self.play(self.lecture[0].animate.set_color(LIME))
        
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "font_size": 24}
        )
        labels = axes.get_axis_labels(x_label="t", y_label="v(t)")
        graph_group = VGroup(axes, labels)
        self.place_in_area(graph_group, "A1", "E6", scale_factor=0.8)
        
        # Velocity function: v(t) = 4 - 0.2*(t-3)^2
        def velocity_func(t):
            return 4 - 0.2 * (t - 3)**2
            
        velocity_curve = axes.plot(velocity_func, x_range=[0, 6], color=LIME)
        
        self.play(Create(axes), Write(labels))
        self.play(Create(velocity_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The area under this curve represents his total displacement.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Show full area briefly to emphasize displacement concept
        full_area_sample = axes.get_area(velocity_curve, x_range=[0, 5], color=YELLOW, opacity=0.3)
        self.play(FadeIn(full_area_sample))
        self.wait(1)
        self.play(FadeOut(full_area_sample))

        # === Animation for Lecture Line 3 ===
        # Imagine a paint roller moving across the time axis.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        roller_x = ValueTracker(0.001)
        
        # A vertical yellow line (#FFFF00) 'roller' moves across the x-axis.
        roller = Line(
            start=axes.c2p(0.001, 0),
            end=axes.c2p(0.001, velocity_func(0.001)),
            color=YELLOW,
            stroke_width=4
        )
        # Persistent roller update
        roller.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(roller_x.get_value(), 0),
            axes.c2p(roller_x.get_value(), velocity_func(roller_x.get_value()))
        ))
        
        # The area under the green curve fills with #FFFF00 behind the roller.
        # Use always_redraw for the filling area
        area_fill = always_redraw(lambda: axes.get_area(
            velocity_curve, 
            x_range=[0, roller_x.get_value()], 
            color=YELLOW, 
            opacity=0.5
        ))
        
        self.add(area_fill, roller)
        self.play(roller_x.animate.set_value(5), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Each sliver of area adds to his total distance.
        self.play(self.lecture[3].animate.set_color(ORANGE))
        
        # The text 'Area = Total Distance' (#FF8C00) fades in prominently.
        dist_text = Text("Area = Total Distance", font_size=32, color=ORANGE)
        # Position inside the graph area
        self.place_at_grid(dist_text, "B3", scale_factor=1.0)
        
        self.play(FadeIn(dist_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Integration sums these infinite tiny slices of progress.
        self.play(self.lecture[4].animate.set_color(WHITE_COLOR))
        
        # The symbol '∫ v(t) dt' (#FFFFFF) appears at the bottom.
        integral_symbol = MathTex(r"\int_{0}^{5} v(t) \, dt", font_size=36, color=WHITE_COLOR)
        # Position at the bottom of the right side grid
        self.place_at_grid(integral_symbol, "F3", scale_factor=1.2)
        
        self.play(Write(integral_symbol))
        self.wait(2)
