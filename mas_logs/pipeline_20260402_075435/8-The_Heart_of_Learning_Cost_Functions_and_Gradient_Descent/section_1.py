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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "The Big Picture: The Blindfolded Hiker"
        lines = [
            "Meet Pixel, a robot starting his learning journey.",
            "Pixel makes a guess for this package's weight.",
            "An error bar shows how far he's off."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Fog background (static visuals)
        fog = VGroup(*[
            Circle(radius=0.5, color=GRAY_E, fill_opacity=0.1, stroke_width=0).move_to(self.grid[f"{r}{c}"])
            for r in ["A", "B", "C", "D", "E", "F"] for c in ["1", "2", "3", "4", "5", "6"]
        ])
        
        # Pixel the Robot (simple shapes)
        pixel_head = Square(side_length=1.0, color=BLUE, fill_opacity=0.8)
        pixel_eye_l = Dot(color=WHITE).move_to(pixel_head.get_center() + LEFT*0.2 + UP*0.2)
        pixel_eye_r = Dot(color=WHITE).move_to(pixel_head.get_center() + RIGHT*0.2 + UP*0.2)
        pixel_mouth = Line(LEFT*0.2, RIGHT*0.2, stroke_width=2, color=WHITE).move_to(pixel_head.get_center() + DOWN*0.2)
        pixel = VGroup(pixel_head, pixel_eye_l, pixel_eye_r, pixel_mouth)
        self.place_at_grid(pixel, "D3", scale_factor=0.8) # Fixed Issue 28: Pixel to D3
        
        # Analogy: Hiker in a Valley
        axes = Axes(x_range=[-2, 2], y_range=[0, 4], x_length=3, y_length=2, 
                    axis_config={"include_tip": False, "stroke_width": 1}).set_color(GRAY)
        valley_curve = axes.plot(lambda x: x**2, color=TEAL)
        valley = VGroup(axes, valley_curve)
        self.place_in_area(valley, "E2", "F5", scale_factor=0.7) # Fixed Issue 30: Use area E2-F5
        
        hiker_dot = Dot(color=YELLOW).move_to(axes.c2p(-1.5, 2.25))
        hiker_label = Text("Hiker", font_size=16, color=YELLOW).next_to(hiker_dot, UP, buff=0.1)
        
        target_dot = Dot(color=GREEN).move_to(axes.c2p(0, 0))
        target_label = Text("Goal: Minimum", font_size=16, color=GREEN).next_to(target_dot, DOWN, buff=0.1)
        
        self.play(FadeIn(fog), FadeIn(pixel), Create(valley), FadeIn(hiker_dot), Write(hiker_label))
        self.play(FadeIn(target_dot), Write(target_label))
        self.play(Indicate(target_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Transition: Clear analogy, focus on prediction
        self.play(FadeOut(valley), FadeOut(hiker_dot), FadeOut(hiker_label), FadeOut(target_dot), FadeOut(target_label))
        
        package = Square(side_length=0.8, color=GOLD_E, fill_opacity=1)
        package_text = Text("5kg", font_size=18, color=BLACK).move_to(package.get_center())
        package_group = VGroup(package, package_text)
        self.place_at_grid(package_group, "A2") # Fixed Issue 29: Package to A2
        
        guess_label = Text("Guess: 10kg", font_size=22, color=BLUE)
        self.place_at_grid(guess_label, "A5") # Fixed Issue 29: Guess to A5
        
        self.play(FadeIn(package_group), Write(guess_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        
        # Visualize the gap/error
        error_line = DoubleArrow(package_group.get_right(), guess_label.get_left(), buff=0.2, color=RED, tip_length=0.15)
        error_text = Text("Error", font_size=20, color=RED).next_to(error_line, UP, buff=0.1)
        
        self.play(Create(error_line), Write(error_text))
        self.wait(2)
