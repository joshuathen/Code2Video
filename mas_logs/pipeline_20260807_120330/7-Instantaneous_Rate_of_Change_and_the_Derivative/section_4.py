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

class Section4Scene(TeachingScene):
    def construct(self):
        # Section Title and Lecture Lines
        title_text = "The Visual Solution: The Shrinking Secant"
        lecture_lines = [
            "Draw a secant line through points A and B.",
            "Next, we slowly slide point B closer to point A.",
            "The gap between them, called 'h', begins to shrink.",
            "As 'h' nears zero, the secant becomes a tangent line.",
            "This tangent represents the speed at that exact moment."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for consistency
        color_a = BLUE_B
        color_b = YELLOW_B
        color_secant = WHITE
        color_h = GREEN_B
        color_tangent = "#FFA500" # Bright Orange

        # 1. Setup Axes and Plot in Area A1-D6 (Issue 43/36)
        axes = Axes(
            x_range=[-0.5, 4.5, 1],
            y_range=[-0.5, 4.5, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_D},
        )
        
        def func(x):
            return 0.2 * x**2 + 0.5

        curve = axes.plot(func, x_range=[0, 4], color=GREY_C)
        
        # Point A (static)
        x_a = 1.0
        pos_a = axes.c2p(x_a, func(x_a))
        dot_a = Dot(pos_a, color=color_a)
        label_a = MathTex("A", color=color_a, font_size=24).next_to(dot_a, UL, buff=0.1)

        # Point B (dynamic)
        x_b_tracker = ValueTracker(3.5)
        dot_b = Dot(color=color_b)
        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_b_tracker.get_value(), func(x_b_tracker.get_value()))))
        
        label_b = MathTex("B", color=color_b, font_size=24)
        label_b.add_updater(lambda l: l.next_to(dot_b, UR, buff=0.1))

        # Main Plot Group
        main_plot_group = VGroup(axes, curve, dot_a, label_a, dot_b, label_b)
        self.place_in_area(main_plot_group, 'A1', 'D6', scale_factor=0.85)

        # 2. Secant Line
        def get_secant_line():
            p1 = dot_a.get_center()
            p2 = dot_b.get_center()
            direction = p2 - p1
            if np.linalg.norm(direction) < 0.001:
                p2 = axes.c2p(x_a + 0.001, func(x_a + 0.001))
                direction = p2 - p1
            
            unit_dir = direction / np.linalg.norm(direction)
            return Line(p1 - unit_dir * 1.5, p1 + unit_dir * 3.0, color=color_secant, stroke_width=2)

        secant_line = always_redraw(get_secant_line)

        # 3. 'h' Gap Indicator in Area E2-F5 (Issue 43/37)
        # Avoid creating MathTex inside always_redraw
        h_line = Line(color=color_h)
        tick_a = Line(UP * 0.1, DOWN * 0.1, color=color_h)
        tick_b = Line(UP * 0.1, DOWN * 0.1, color=color_h)
        h_text = MathTex("h", color=color_h, font_size=24)
        
        h_group = VGroup(h_line, tick_a, tick_b, h_text)
        
        y_base = self.grid['E2'][1] # Use Row E for horizontal gap line

        def update_h_group(obj):
            x_a_grid = dot_a.get_center()[0]
            x_b_grid = dot_b.get_center()[0]
            # Update line
            obj[0].set_points_as_corners([[x_a_grid, y_base, 0], [x_b_grid, y_base, 0]])
            # Update ticks
            obj[1].move_to([x_a_grid, y_base, 0])
            obj[2].move_to([x_b_grid, y_base, 0])
            # Update text
            obj[3].next_to(obj[0], DOWN, buff=0.1)

        h_group.add_updater(update_h_group)

        # === Animation for Lecture Line 1 ===
        # "Draw a secant line through points A and B."
        self.lecture[0].set_color(color_secant)
        self.play(Create(axes), Create(curve), run_time=1)
        self.play(FadeIn(dot_a, label_a), FadeIn(dot_b, label_b))
        self.play(Create(secant_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Next, we slowly slide point B closer to point A."
        self.lecture[1].set_color(color_b)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The gap between them, called 'h', begins to shrink."
        self.lecture[2].set_color(color_h)
        self.play(FadeIn(h_group))
        
        # Slide B towards A
        self.play(
            x_b_tracker.animate.set_value(x_a + 0.05),
            run_time=6,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "As 'h' nears zero, the secant becomes a tangent line."
        self.lecture[3].set_color(color_tangent)
        
        # Switch to tangent line
        current_secant = secant_line.copy()
        self.remove(secant_line)
        self.add(current_secant)
        
        # Final point B position
        x_b_tracker.set_value(x_a)
        
        # Define Tangent Line
        p1 = dot_a.get_center()
        p_close = axes.c2p(x_a + 0.01, func(x_a + 0.01))
        tan_dir = p_close - p1
        tan_unit = tan_dir / np.linalg.norm(tan_dir)
        tangent_line = Line(p1 - tan_unit * 2.0, p1 + tan_unit * 3.5, color=color_tangent, stroke_width=4)
        
        self.play(
            Transform(current_secant, tangent_line),
            FadeOut(h_group),
            FadeOut(dot_b, label_b),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This tangent represents the speed at that exact moment."
        self.lecture[4].set_color(color_tangent)
        self.play(Indicate(current_secant, color=color_tangent, scale_factor=1.05))
        self.wait(2)
