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
        title = "The Heat Equation: Diffusion in Action"
        lines = [
            "The Heat Equation describes how temperature spreads over time.",
            "It follows the Curvature Principle of local diffusion.",
            "Hotter spots with 'concave down' curves lose heat.",
            "Colder 'concave up' regions gain heat from neighbors.",
            "The rate of change depends on the local curvature."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_EQ = "#FFFF00"
        COLOR_HOT = "#FF0000"
        COLOR_COLD = "#0000FF"
        COLOR_NEUTRAL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_EQ))
        
        eq = MathTex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u", color=COLOR_EQ)
        # Resolved Issue 36: Equation size and position
        self.place_in_area(eq, 'A3', 'B6', scale_factor=0.9)
        self.play(Write(eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_NEUTRAL)
        )
        
        # Setup Axes and Curve
        # Resolved Issue 37: Axes position to avoid lecture overlap
        axes = Axes(
            x_range=[0, 2 * PI, PI],
            y_range=[0, 4, 1],
            x_length=4.0,
            y_length=2.2,
            axis_config={"include_tip": True, "color": GREY_A},
            tips=False
        )
        self.place_in_area(axes, 'C3', 'F6', scale_factor=0.8)
        
        alpha_val = 0.4
        time_tracker = ValueTracker(0)
        
        # Curve defined with updater for flattening effect
        curve = axes.plot(lambda x: np.sin(x) + 2, color=WHITE)
        curve.add_updater(lambda m: m.become(
            axes.plot(lambda x: np.sin(x) * np.exp(-alpha_val * time_tracker.get_value()) + 2, color=WHITE)
        ))
        
        rod = Line(
            start=axes.c2p(0, 0),
            end=axes.c2p(2*PI, 0),
            stroke_width=6,
            color=GREY_B
        )
        
        rod_label = Text("1D Rod", font_size=16, color=GREY_B)
        # Resolved Issue 38: Rod label position
        self.place_at_grid(rod_label, 'F5', scale_factor=0.8)
        
        self.play(Create(axes), Create(rod), FadeIn(rod_label))
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HOT)
        )
        
        # Highlight peak (concave down)
        peak_dot = Dot(color=COLOR_HOT)
        peak_dot.add_updater(lambda d: d.move_to(
            axes.c2p(PI/2, np.sin(PI/2) * np.exp(-alpha_val * time_tracker.get_value()) + 2)
        ))
        
        peak_label = Text("Concave Down", font_size=14, color=COLOR_HOT)
        peak_label.add_updater(lambda l: l.next_to(peak_dot, UP, buff=0.1))
        
        arrow_down = Arrow(start=UP*0.4, end=ORIGIN, color=COLOR_HOT, buff=0).scale(0.5)
        arrow_down.add_updater(lambda a: a.next_to(peak_dot, DOWN, buff=0.1))
        
        self.play(FadeIn(peak_dot), FadeIn(peak_label), FadeIn(arrow_down))
        # Show gradual drop
        self.play(time_tracker.animate.set_value(1.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_COLD)
        )
        
        # Highlight valley (concave up)
        valley_dot = Dot(color=COLOR_COLD)
        valley_dot.add_updater(lambda d: d.move_to(
            axes.c2p(3*PI/2, np.sin(3*PI/2) * np.exp(-alpha_val * time_tracker.get_value()) + 2)
        ))
        
        valley_label = Text("Concave Up", font_size=14, color=COLOR_COLD)
        valley_label.add_updater(lambda l: l.next_to(valley_dot, DOWN, buff=0.1))
        
        arrow_up = Arrow(start=DOWN*0.4, end=ORIGIN, color=COLOR_COLD, buff=0).scale(0.5)
        arrow_up.add_updater(lambda a: a.next_to(valley_dot, UP, buff=0.1))
        
        self.play(FadeIn(valley_dot), FadeIn(valley_label), FadeIn(arrow_up))
        # Continue thermal equalization
        self.play(time_tracker.animate.set_value(3.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_NEUTRAL)
        )
        
        # Final flattening out
        self.play(
            time_tracker.animate.set_value(7),
            FadeOut(peak_dot), FadeOut(peak_label), FadeOut(arrow_down),
            FadeOut(valley_dot), FadeOut(valley_label), FadeOut(arrow_up),
            run_time=3
        )
        
        self.wait(2)
