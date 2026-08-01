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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Formal Language: The Epsilon-Delta Game",
            [
                'How do we define "getting closer" with absolute precision?',
                'A challenger sets an epsilon error tolerance on y.',
                'We must find a delta distance on the x-axis.',
                'All points within delta must land inside the epsilon-target.',
                "This formal game defines the limit's existence."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display a white (#FFFFFF) curve and a point 'L' marked on the vertical y-axis.
        
        # Define axes and curve
        # Origin at center of right side area (approx 3.5, -0.3)
        axes = Axes(
            x_range=[-2, 3],
            y_range=[-1, 4],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, 'A2', 'F6')

        # Function f(x) = 0.5(x-1)^2 + 1.5, c=1, L=1.5
        c_val = 1
        l_val = 1.5
        f = lambda x: 0.5 * (x - c_val)**2 + l_val
        curve = axes.plot(f, x_range=[-1.5, 2.5], color=WHITE)
        
        # Points on axes
        l_point = Dot(axes.c2p(0, l_val), color=WHITE)
        c_point = Dot(axes.c2p(c_val, 0), color=WHITE)
        
        # Labels using grid
        l_label = Text("L", font_size=24, color=WHITE)
        self.place_at_grid(l_label, "B3") # (2.5, 1.2) near dot (3.5, 1.2)
        
        c_label = Text("c", font_size=24, color=WHITE)
        self.place_at_grid(c_label, "D5") # (4.5, -0.8) near dot (4.5, -0.3)

        self.play(
            Create(axes),
            Create(curve),
            FadeIn(l_point),
            FadeIn(c_point),
            Write(l_label),
            Write(c_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Two horizontal dashed blue lines (#3388FF) at 'L + epsilon' and 'L - epsilon'
        
        self.play(self.lecture[1].animate.set_color("#3388FF"))
        
        eps_tracker = ValueTracker(1.2)
        h_line_up = always_redraw(lambda: DashedLine(
            axes.c2p(-2, l_val + eps_tracker.get_value()), 
            axes.c2p(3, l_val + eps_tracker.get_value()), 
            color="#3388FF"
        ))
        h_line_down = always_redraw(lambda: DashedLine(
            axes.c2p(-2, l_val - eps_tracker.get_value()), 
            axes.c2p(3, l_val - eps_tracker.get_value()), 
            color="#3388FF"
        ))
        
        eps_label = always_redraw(lambda: Text(
            f"L + ε", font_size=16, color="#3388FF"
        ).move_to(axes.c2p(-1.5, l_val + eps_tracker.get_value() + 0.2)))

        self.play(Create(h_line_up), Create(h_line_down), Write(eps_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Two vertical dashed orange lines (#FFBB33) at 'c + delta' and 'c - delta'
        
        self.play(self.lecture[2].animate.set_color("#FFBB33"))
        
        del_tracker = ValueTracker(1.2)
        v_line_left = always_redraw(lambda: DashedLine(
            axes.c2p(c_val - del_tracker.get_value(), -1), 
            axes.c2p(c_val - del_tracker.get_value(), 4), 
            color="#FFBB33"
        ))
        v_line_right = always_redraw(lambda: DashedLine(
            axes.c2p(c_val + del_tracker.get_value(), -1), 
            axes.c2p(c_val + del_tracker.get_value(), 4), 
            color="#FFBB33"
        ))

        self.play(Create(v_line_left), Create(v_line_right))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Shrink the blue band; then shrink the orange band until the function segment turns green (#00FF00)
        
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # Shrink epsilon
        self.play(eps_tracker.animate.set_value(0.4), run_time=2)
        self.wait(0.5)
        # Shrink delta
        self.play(del_tracker.animate.set_value(0.5), run_time=2)
        
        # Highlight segment green
        green_segment = axes.plot(f, x_range=[c_val - 0.5, c_val + 0.5], color="#00FF00")
        self.play(Create(green_segment))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The intersection area of the bands pulses white (#FFFFFF)
        
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Create intersection rectangle based on final trackers
        rect_width = abs(axes.c2p(c_val + 0.5, 0)[0] - axes.c2p(c_val - 0.5, 0)[0])
        rect_height = abs(axes.c2p(0, l_val + 0.4)[1] - axes.c2p(0, l_val - 0.4)[1])
        
        pulse_rect = Rectangle(
            width=rect_width,
            height=rect_height,
            fill_color=WHITE,
            fill_opacity=0.4,
            stroke_width=0
        ).move_to(axes.c2p(c_val, l_val))
        
        self.play(FadeIn(pulse_rect))
        self.play(pulse_rect.animate.scale(1.2), rate_func=there_and_back, run_time=1)
        self.play(pulse_rect.animate.scale(1.2), rate_func=there_and_back, run_time=1)
        self.play(FadeOut(pulse_rect))
        
        self.wait(2)
