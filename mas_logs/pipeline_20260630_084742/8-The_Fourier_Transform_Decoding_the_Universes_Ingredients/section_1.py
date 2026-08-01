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
        lecture_lines = [
            'Imagine a complex smoothie with many mixed flavors.',
            'The Fourier Transform acts like a mathematical prism.',
            'It separates the smoothie into individual ingredients.'
        ]
        self.setup_layout("The Big Idea: The Smoothie Metaphor", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(PURPLE))
        
        # Create a complex wave
        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": GREY}
        )
        
        def complex_wave_func(x):
            return 1.0 * np.sin(x) + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x)
        
        complex_plot = axes.plot(complex_wave_func, color="#A020F0")
        complex_group = VGroup(axes, complex_plot)
        # Fix for Issue 35: Move to area B1-E4
        self.place_in_area(complex_group, "B1", "E4", scale_factor=1.0)
        
        self.play(Create(axes), Create(complex_plot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Prism icon (Triangle)
        prism = Triangle(color=WHITE, fill_opacity=0.3).scale(0.8)
        self.place_in_area(prism, "C5", "D6")
        
        self.play(Create(prism))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )

        # Create three smaller axes and waves
        # Red wave (Strawberry)
        axes_red = Axes(x_range=[0, 4*PI], y_range=[-1.5, 1.5], x_length=4, y_length=1, axis_config={"include_tip": False, "color": GREY})
        plot_red = axes_red.plot(lambda x: 1.0 * np.sin(x), color="#FF0000")
        label_red = Text("3 Strawberries", font_size=16, color="#FF0000")
        group_red = VGroup(axes_red, plot_red)
        
        # Yellow wave (Banana)
        axes_yellow = Axes(x_range=[0, 4*PI], y_range=[-1.5, 1.5], x_length=4, y_length=1, axis_config={"include_tip": False, "color": GREY})
        plot_yellow = axes_yellow.plot(lambda x: 0.5 * np.sin(3 * x), color="#FFFF00")
        label_yellow = Text("1 Banana", font_size=16, color="#FFFF00")
        group_yellow = VGroup(axes_yellow, plot_yellow)
        
        # Blue wave (Blueberry)
        axes_blue = Axes(x_range=[0, 4*PI], y_range=[-1.5, 1.5], x_length=4, y_length=1, axis_config={"include_tip": False, "color": GREY})
        plot_blue = axes_blue.plot(lambda x: 0.2 * np.sin(5 * x), color="#0000FF")
        label_blue = Text("10 Blueberries", font_size=16, color="#0000FF")
        group_blue = VGroup(axes_blue, plot_blue)

        # Positioning
        self.place_in_area(group_red, "A1", "B4", scale_factor=0.8)
        # Fix for Issue 36: Use area for label_red
        self.place_in_area(label_red, "B5", "B6", scale_factor=0.8)
        
        self.place_in_area(group_yellow, "C1", "D4", scale_factor=0.8)
        self.place_in_area(label_yellow, "D5", "D6", scale_factor=0.8)
        
        self.place_in_area(group_blue, "E1", "F4", scale_factor=0.8)
        # Fix for Issue 37: Use area for label_blue
        self.place_in_area(label_blue, "F5", "F6", scale_factor=0.8)

        # Transition: Complex wave passes through prism and separates
        self.play(
            complex_group.animate.move_to(prism.get_center()).scale(0.1).set_opacity(0),
            run_time=1.5
        )
        self.play(
            FadeOut(prism),
            FadeIn(group_red),
            FadeIn(label_red),
            FadeIn(group_yellow),
            FadeIn(label_yellow),
            FadeIn(group_blue),
            FadeIn(label_blue)
        )
        self.wait(2)
