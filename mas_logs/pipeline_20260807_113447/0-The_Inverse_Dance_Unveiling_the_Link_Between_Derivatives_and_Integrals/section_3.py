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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Area Function (The Fundamental Link)"
        lines = [
            "Define the Area Function A(x).",
            "It tracks area from 'a' to 'x'.",
            "As 'x' moves, the area grows.",
            "Let's visualize this moving boundary.",
            "This function bridges slopes and areas."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        BLUE_COLOR = "#0000FF"
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Define the Area Function A(x).
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Define Axes (C1-F5 as requested by Issue 35)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 3, 1],
            x_length=4.5,
            y_length=3.0,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, 'C1', 'F5')
        
        # Function f(t) = 0.1*t^2 + 0.5
        func = axes.plot(lambda t: 0.1 * t**2 + 0.5, x_range=[0, 4.5], color=WHITE)
        func_label = MathTex("f(t)", color=WHITE, font_size=24)
        # Position func label at B5 as requested by Issue 36
        self.place_at_grid(func_label, 'B5')
        
        # Fixed point 'a' on x-axis
        a_val = 1.0
        a_line = axes.get_vertical_line(axes.c2p(a_val, func.underlying_function(a_val)), color=GRAY)
        a_label = MathTex("a", color=WHITE, font_size=24)
        a_label.move_to(axes.c2p(a_val, -0.3))
        
        self.play(Create(axes), Create(func), Write(func_label))
        self.play(Create(a_line), Write(a_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It tracks area from 'a' to 'x'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        x_val_tracker = ValueTracker(2.0)
        
        # Shaded region from 'a' to 'x' in blue (#0000FF)
        area = always_redraw(lambda: 
            axes.get_area(func, x_range=[a_val, x_val_tracker.get_value()], color=BLUE_COLOR, opacity=0.5)
        )
        
        # Moving vertical line at 'x'
        x_line = Line(color=WHITE)
        def update_x_line(m):
            curr_x = x_val_tracker.get_value()
            curr_y = func.underlying_function(curr_x)
            m.set_points_as_corners([axes.c2p(curr_x, 0), axes.c2p(curr_x, curr_y)])
        x_line.add_updater(update_x_line)
        
        # Moving label for 'x'
        x_label = MathTex("x", color=WHITE, font_size=24)
        def update_x_label(m):
            m.move_to(axes.c2p(x_val_tracker.get_value(), -0.3))
        x_label.add_updater(update_x_label)
        
        # Label for the area function A(x) in blue (#0000FF)
        area_label = MathTex("A(x)", color=BLUE_COLOR, font_size=32)
        # Place near the initial area
        self.place_at_grid(area_label, "B3")
        
        self.play(FadeIn(area), Create(x_line), Write(x_label), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # As 'x' moves, the area grows.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Animate the boundary moving to show growth
        self.play(x_val_tracker.animate.set_value(4.0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Let's visualize this moving boundary (Snow-Plow icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/snowpl.svg]).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Load Snow-Plow asset (Issue 27)
        plow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowpl.svg")
        plow.scale(0.3)
        
        def update_plow(m):
            curr_x = x_val_tracker.get_value()
            curr_y = func.underlying_function(curr_x)
            # Position the plow on the moving line 'x', centered vertically on the boundary
            m.move_to(axes.c2p(curr_x, curr_y / 2))
        
        plow.add_updater(update_plow)
        self.add(plow)
        
        # Move the boundary back and forth to emphasize the accumulation process
        self.play(x_val_tracker.animate.set_value(1.5), run_time=2)
        self.play(x_val_tracker.animate.set_value(3.8), run_time=3)
        self.wait(1)
        
        # Remove the plow effect
        plow.remove_updater(update_plow)
        self.play(FadeOut(plow))

        # === Animation for Lecture Line 5 ===
        # This function bridges slopes and areas.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Formal definition of the Area Function in white (#FFFFFF)
        formula = MathTex(r"A(x) = \int_a^x f(t) \, dt", color=WHITE, font_size=36)
        # Position the formula in area A1-A5 (Issue 37)
        self.place_in_area(formula, 'A1', 'A5')
        
        self.play(Write(formula))
        self.wait(2)
