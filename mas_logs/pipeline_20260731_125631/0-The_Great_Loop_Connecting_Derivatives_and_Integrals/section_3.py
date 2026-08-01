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
        # Data from storyboard
        title_text = "The Accumulation Function"
        lecture_lines = [
            "Let's build a function for the shaded area.",
            "The area grows as the slider moves.",
            "This moving area is itself a function.",
            "For $f(t) = 2t$, the area is $x^2$.",
            "A new curve emerges from the old."
        ]
        
        # Setup Layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"
        AREA_COLOR = "#4169E1"
        F_COLOR = WHITE
        S_COLOR = "#FFFF00"
        
        # Tracker for slider position x
        x_tracker = ValueTracker(0.01) # Start slightly above 0 to avoid degenerate geometry

        # === Animation for Lecture Line 1 ===
        # Plot f(t) = 2t as a white line
        axes_f = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 8, 2],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "font_size": 24}
        ).add_coordinates()
        
        label_f = MathTex(r"f(t) = 2t", font_size=28, color=F_COLOR)
        
        # Group and place in the top area (A1 to C6)
        # Apply fix from Issue 24: self.place_in_area(f_group, 'A2', 'C6', scale_factor=0.8)
        f_group = VGroup(axes_f, label_f)
        self.place_in_area(f_group, 'A2', 'C6', scale_factor=0.8)
        # Re-align label relative to axes after scaling/moving
        label_f.next_to(axes_f, UP, buff=0.1)
        
        graph_f = axes_f.plot(lambda t: 2*t, x_range=[0, 3.5], color=F_COLOR)
        
        self.play(
            Write(axes_f),
            Write(label_f),
            Create(graph_f),
            self.lecture[0].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The area grows as the slider moves.
        
        # Slider vertical line spanning from axis to the function
        slider = always_redraw(lambda: Line(
            axes_f.c2p(x_tracker.get_value(), 0),
            axes_f.c2p(x_tracker.get_value(), 2 * x_tracker.get_value()),
            color=WHITE, stroke_width=4
        ))
        
        # Shaded area VMobject using updater for efficiency
        area_poly = VMobject()
        area_poly.set_fill(AREA_COLOR, opacity=0.5)
        area_poly.set_stroke(width=0)
        
        def update_area(poly):
            val = x_tracker.get_value()
            poly.set_points_as_corners([
                axes_f.c2p(0, 0),
                axes_f.c2p(val, 0),
                axes_f.c2p(val, 2*val)
            ])
        
        area_poly.add_updater(update_area)
        
        self.add(area_poly, slider)
        self.play(
            x_tracker.animate.set_value(1.5),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # This moving area is itself a function.
        self.play(
            x_tracker.animate.set_value(3.0),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A second set of axes appears below; points are plotted.
        axes_s = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 10, 2],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "font_size": 24}
        ).add_coordinates()
        
        label_s = MathTex(r"S(x) = \int_0^x f(t) \, dt", font_size=28, color=S_COLOR)
        
        # Group and place in the bottom area (D1 to F6)
        # Apply fix from Issue 25: self.place_in_area(s_group, 'D2', 'F6', scale_factor=0.8)
        s_group = VGroup(axes_s, label_s)
        self.place_in_area(s_group, 'D2', 'F6', scale_factor=0.8)
        label_s.next_to(axes_s, UP, buff=0.1)
        
        # Dot on S(x) graph representing accumulated area
        dot = Dot(color=S_COLOR)
        dot.add_updater(lambda d: d.move_to(axes_s.c2p(x_tracker.get_value(), x_tracker.get_value()**2)))
        
        # Sync back to show the mapping process from the start
        self.play(
            Write(axes_s),
            Write(label_s),
            FadeIn(dot),
            x_tracker.animate.set_value(0.01),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # The points on the lower graph connect to form the parabola S(x) = x^2 in yellow.
        graph_s = axes_s.plot(lambda x: x**2, x_range=[0, 3], color=S_COLOR)
        
        label_s_final = MathTex(r"S(x) = x^2", font_size=24, color=S_COLOR)
        # Position label relative to the end of the curve
        label_s_final.next_to(axes_s.c2p(3, 9), UR, buff=0.1)
        
        self.play(
            x_tracker.animate.set_value(3.0),
            Create(graph_s),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR),
            run_time=3,
            rate_func=linear
        )
        self.play(Write(label_s_final))
        
        self.wait(3)
