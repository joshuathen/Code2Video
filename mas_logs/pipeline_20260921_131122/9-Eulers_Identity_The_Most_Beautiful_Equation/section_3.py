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
            "Euler's Formula relates exponentials to trig functions.",
            "e to the ix equals cos x plus i sin x.",
            "This formula describes motion along the unit circle.",
            "As x grows, the point traverses the circumference.",
            "This connects the exponential and trigonometric worlds."
        ]
        self.setup_layout("Deriving Euler’s Formula", lecture_lines)
        
        colors = [BLUE, GREEN, YELLOW, ORANGE, PURPLE]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        eq1 = MathTex("e^x = \\sum \\frac{x^n}{n!}", font_size=36)
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        group1 = VGroup(eq1, icon1).arrange(RIGHT)
        self.place_in_area(group1, 'A3', 'B4', scale_factor=0.9)
        self.play(Write(eq1), FadeIn(icon1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        eq2 = MathTex("e^{i\\theta} = \\cos(\\theta) + i\\sin(\\theta)", font_size=36)
        self.place_in_area(eq2, 'C3', 'D4', scale_factor=0.9)
        self.play(Write(eq2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        axes = Axes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], axis_config={"include_tip": False})
        circle = Circle(radius=1, color=WHITE)
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        unit_circle_group = VGroup(axes, circle, icon2).arrange(DOWN)
        self.place_in_area(unit_circle_group, 'E2', 'F5', scale_factor=0.7)
        self.play(Create(axes), Create(circle), FadeIn(icon2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        point = Dot(color=RED)
        tracker = ValueTracker(0)
        point.add_updater(lambda m: m.move_to(axes.c2p(np.cos(tracker.get_value()), np.sin(tracker.get_value()))))
        self.add(point)
        self.play(tracker.animate.set_value(2*PI), run_time=3, rate_func=linear)
        point.remove_updater(lambda m: m.move_to(axes.c2p(np.cos(tracker.get_value()), np.sin(tracker.get_value()))))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        connect_line = Line(eq2.get_bottom(), unit_circle_group.get_top(), color=WHITE)
        self.play(Create(connect_line))
        self.wait(2)
