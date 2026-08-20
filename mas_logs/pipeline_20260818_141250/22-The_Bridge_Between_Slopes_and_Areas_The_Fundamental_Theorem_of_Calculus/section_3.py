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
        lecture_lines = [
            "Derivative of accumulation is original.",
            "Filling tank rate matches flow.",
            "Linking change to total area.",
            "Formally: d/dx integral is f(x).",
            "This is fundamental theorem part one."
        ]
        self.setup_layout("Fundamental Theorem of Calculus (Part 1)", lecture_lines)
        
        # Create graphical elements
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], x_length=4, y_length=2.5, axis_config={"include_tip": False})
        f_func = ax.plot(lambda x: 0.25 * x**2 + 0.5, color=BLUE)
        f_label = MathTex("f(t)").next_to(f_func, UP)
        
        # Use persistent mobject for area with updater
        area = ax.get_area(f_func, x_range=[0, 2], color=GREEN, opacity=0.3)
        x_tracker = ValueTracker(2)
        
        def update_area(mob):
            new_area = ax.get_area(f_func, x_range=[0, x_tracker.get_value()], color=GREEN, opacity=0.3)
            mob.become(new_area)
            
        area.add_updater(update_area)
        
        graph_group = VGroup(ax, f_func, f_label, area)
        self.place_in_area(graph_group, 'A4', 'D6', scale_factor=0.6)
        
        eq = MathTex(r"F(x) = \int_{a}^{x} f(t) \, dt").scale(0.8)
        self.place_at_grid(eq, 'E4', scale_factor=0.75)
        
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")
        self.place_at_grid(tank, 'B1', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Write(graph_group), FadeIn(tank), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW), Create(eq), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN), x_tracker.animate.set_value(3.5), run_time=2)

        # === Animation for Lecture Line 4 ===
        d_eq = MathTex(r"\frac{d}{dx} \int_{a}^{x} f(t) dt = f(x)").scale(0.8)
        self.place_at_grid(d_eq, 'F4', scale_factor=0.75)
        self.play(self.lecture[3].animate.set_color(RED), FadeIn(d_eq), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE), run_time=1)
        self.wait(2)
