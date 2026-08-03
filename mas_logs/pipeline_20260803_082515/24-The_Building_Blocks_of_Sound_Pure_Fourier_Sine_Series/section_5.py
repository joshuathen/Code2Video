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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Construction Site"
        lecture_lines = [
            "- We start with the smooth, fundamental sine wave.",
            "- Adding the third harmonic begins to flatten the peaks.",
            "- The fifth and seventh harmonics sharpen the vertical edges.",
            "- As we add more, the 'wiggles' settle into blocks.",
            "- Infinite smooth waves eventually build perfect sharp corners."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors for matching lecture lines
        COLOR_FUNDAMENTAL = "#00FF00"  # Green
        COLOR_H3 = "#00FFFF"           # Cyan
        COLOR_H_ALL = "#FFFFFF"        # White
        COLOR_CORNERS = "#FFFF00"      # Yellow

        # Axes setup (placed in area A1 to F6)
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, 'A1', 'F6')
        self.add(axes)

        # Fourier summation function: f(x) = sum_{n=1,3,...}^N (1/n) * sin(nx)
        def fourier_sum(x, n_max):
            return sum([(1/n) * np.sin(n * x) for n in range(1, n_max + 1, 2)])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_FUNDAMENTAL)
        
        current_wave = axes.plot(lambda x: np.sin(x), color=COLOR_FUNDAMENTAL)
        n_label = MathTex("n=1", color=COLOR_FUNDAMENTAL, font_size=28)
        self.place_at_grid(n_label, "A5", scale_factor=0.8)

        self.play(Create(current_wave), Write(n_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_H3)

        # Visualize the 3rd harmonic before summing
        h3_wave = axes.plot(lambda x: (1/3) * np.sin(3*x), color=COLOR_H3)
        h3_label = MathTex("n=3", color=COLOR_H3, font_size=28)
        self.place_at_grid(h3_label, "B5", scale_factor=0.8)
        
        sum_wave_n3 = axes.plot(lambda x: fourier_sum(x, 3), color=COLOR_H3)
        
        self.play(Create(h3_wave), Write(h3_label), run_time=1.5)
        self.wait(0.5)
        self.play(
            FadeOut(h3_wave),
            Transform(current_wave, sum_wave_n3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_H_ALL)

        # Advance to 7 harmonics
        sum_wave_n7 = axes.plot(lambda x: fourier_sum(x, 7), color=COLOR_H_ALL)
        n_label_7 = MathTex("n=1...7", color=COLOR_H_ALL, font_size=28)
        self.place_at_grid(n_label_7, "A5", scale_factor=0.8)

        self.play(
            FadeOut(n_label),
            FadeOut(h3_label),
            Transform(current_wave, sum_wave_n7),
            Write(n_label_7),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_H_ALL)

        # Rapidly add harmonics to approach a square wave
        n_steps = [11, 15, 21, 31, 51]
        for n_val in n_steps:
            next_wave = axes.plot(lambda x: fourier_sum(x, n_val), color=COLOR_H_ALL)
            self.play(Transform(current_wave, next_wave), run_time=0.4)
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_CORNERS)

        # Highlight corners to emphasize sharpness
        corner_highlight_1 = Circle(radius=0.25, color=COLOR_CORNERS).move_to(axes.coords_to_point(PI/2, 0.8))
        corner_highlight_2 = Circle(radius=0.25, color=COLOR_CORNERS).move_to(axes.coords_to_point(-PI/2, -0.8))
        
        sharp_text = Text("Sharp Corners", color=COLOR_CORNERS, font_size=20)
        self.place_at_grid(sharp_text, "E6", scale_factor=0.8)

        self.play(
            Create(corner_highlight_1), 
            Create(corner_highlight_2), 
            Write(sharp_text),
            run_time=2
        )
        self.wait(3)
