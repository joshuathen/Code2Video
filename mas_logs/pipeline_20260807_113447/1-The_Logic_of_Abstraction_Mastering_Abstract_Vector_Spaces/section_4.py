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
        # Define colors for lecture lines and corresponding visual elements
        COLOR_1 = RED_B
        COLOR_2 = BLUE_B
        COLOR_SUM = PURPLE_B
        
        self.setup_layout("Visualizing Function Spaces", [
            "Continuous functions behave like vectors.",
            "Summing curves produces a new curve.",
            "Scaling curves changes their amplitude."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Initialize Axes for the right side
        # Placing the coordinate system in a centered right area (B3 to E6)
        axes = Axes(
            x_range=[-2.2, 2.2, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": False, "color": GRAY_C}
        )
        self.place_in_area(axes, 'B3', 'E6')
        
        # Define base functions
        f_wave = axes.plot(lambda x: np.sin(PI * x), color=COLOR_1)
        g_wave = axes.plot(lambda x: 0.5 * np.cos(2 * PI * x), color=COLOR_2)
        
        # Labels for the base functions
        # Positioned slightly offset to avoid overlapping the axis
        f_label = MathTex("f(x)", color=COLOR_1, font_size=24)
        g_label = MathTex("g(x)", color=COLOR_2, font_size=24)
        self.place_at_grid(f_label, 'B6', scale_factor=1.0)
        self.place_at_grid(g_label, 'C6', scale_factor=1.0)

        self.play(Create(axes))
        self.play(Create(f_wave), FadeIn(f_label))
        self.play(Create(g_wave), FadeIn(g_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Shift focus to summing curves
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_SUM)
        )
        
        # Resultant function h(x) = f(x) + g(x)
        h_func = lambda x: np.sin(PI * x) + 0.5 * np.cos(2 * PI * x)
        h_wave = axes.plot(h_func, color=COLOR_SUM)
        h_label = MathTex("h(x) = f(x) + g(x)", color=COLOR_SUM, font_size=24)
        self.place_at_grid(h_label, 'A5', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(f_wave, h_wave),
            ReplacementTransform(g_wave, h_wave),
            FadeOut(f_label),
            FadeOut(g_label),
            FadeIn(h_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Shift focus to scaling amplitude
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SUM)
        )
        
        # Value tracker for amplitude scaling
        amp_tracker = ValueTracker(1.0)
        
        # Create a dynamic plot using always_redraw for the amplitude change
        # Note: B011 warns about expensive mobjects in always_redraw, 
        # but a simple parametric curve (plot) is standard practice for this level of detail.
        scaled_h_wave = always_redraw(lambda: axes.plot(
            lambda x: amp_tracker.get_value() * h_func(x),
            color=COLOR_SUM
        ))
        
        # Replace static wave with the dynamic one
        self.remove(h_wave)
        self.add(scaled_h_wave)
        
        # Animate amplitude scaling up and down
        self.play(amp_tracker.animate.set_value(1.8), run_time=1.5, rate_func=smooth)
        self.play(amp_tracker.animate.set_value(0.4), run_time=1.5, rate_func=smooth)
        self.play(amp_tracker.animate.set_value(1.0), run_time=1, rate_func=smooth)
        
        self.wait(3)
