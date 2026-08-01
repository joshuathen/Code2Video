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
        title_text = "The Riemann Hypothesis & The Critical Line"
        lecture_lines = [
            "Mathematicians search for points where the output is zero.",
            "These \"zeros\" hold the secret to prime distribution.",
            "Most zeros lie on a single vertical critical line.",
            "The Riemann Hypothesis claims *all* non-trivial zeros are there.",
            "Solving this mystery unlocks the DNA of prime numbers."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        YELLOW_COLOR = "#FFFF00"
        BLUE_COLOR = "#00FFFF"
        RED_COLOR = "#FF0000"
        GREY_COLOR = "#888888"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_COLOR)
        # Background grid/axes for context
        v_axis = Line(self.grid["A3"], self.grid["F3"], color=GREY_COLOR, stroke_opacity=0.3)
        h_axis = Line(self.grid["C1"], self.grid["C6"], color=GREY_COLOR, stroke_opacity=0.3)
        self.add(v_axis, h_axis)
        
        # Show some random dots as potential zeros
        initial_dots = VGroup(*[Dot(color=BLUE_COLOR, radius=0.06) for _ in range(3)])
        self.place_at_grid(initial_dots[0], "B2")
        self.place_at_grid(initial_dots[1], "D4")
        self.place_at_grid(initial_dots[2], "E2") # Fixed Issue 37: Moved from E1 to E2
        self.play(FadeIn(initial_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_COLOR)
        # Zeros pulsing to show importance
        self.play(
            *[z.animate.scale(1.5) for z in initial_dots],
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW_COLOR)
        
        # Remove initial dots to focus on the line
        self.play(FadeOut(initial_dots))
        
        # Animation 1: Focus on vertical line at Re(s) = 0.5 in bright yellow (#FFFF00).
        # We extend it slightly beyond the grid for the pan
        critical_line = Line(self.grid["A3"] + UP*2, self.grid["F3"] + DOWN*2, color=YELLOW_COLOR, stroke_width=4)
        
        # Animation 2: Label 'Critical Line' appears next to the yellow line.
        critical_label = Text("Critical Line", font_size=20, color=YELLOW_COLOR)
        self.place_at_grid(critical_label, "B5", scale_factor=0.8) # Fixed Issue 38: Moved from B4 to B5, scaled to 0.8
        
        self.play(Create(critical_line))
        self.play(Write(critical_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE_COLOR)
        
        # Animation 3: Blue points (zeros) blink into existence along the critical line (#00FFFF).
        zeros = VGroup()
        for i in range(15):
            z = Dot(color=BLUE_COLOR, radius=0.07)
            # Position along column 3 (x=2.5). Spread them out.
            y_val = 3.5 - (i * 0.6) 
            z.move_to([2.5, y_val, 0])
            zeros.add(z)
            
        self.play(LaggedStart(*[FadeIn(z) for z in zeros], lag_ratio=0.15))
        
        # Animation 4: Camera pans up the line showing a sequence of zeros.
        # We simulate the pan by shifting the line and zeros down.
        moving_group = VGroup(critical_line, critical_label, zeros, v_axis)
        
        self.play(moving_group.animate.shift(DOWN * 2.5), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(RED_COLOR)
        
        # Animation 5: Large question mark (#FF0000) pulse-flashes over the critical line.
        question_mark = Text("?", font_size=140, color=RED_COLOR)
        # Place it centrally in the visual area
        self.place_in_area(question_mark, "C5", "F6", scale_factor=0.8) # Fixed Issue 39: Area moved to C5-F6, scale to 0.8
        
        self.play(FadeIn(question_mark))
        # Manual pulse-flash loop
        for _ in range(3):
            self.play(
                question_mark.animate.scale(1.15).set_opacity(0.6),
                rate_func=there_and_back,
                run_time=0.4
            )
        
        self.wait(2)
