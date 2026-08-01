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
        self.setup_layout("The Messy Signal (Superposition)", [
            "Real-world signals combine multiple sine waves together.",
            "Summing different frequencies creates a messy, complex wave.",
            "This \"Time Domain\" view hides the individual components."
        ])

        # Colors
        RED_C = "#E74C3C"
        GREEN_C = "#2ECC71"
        BLUE_C = "#3498DB"
        WHITE_C = "#ECF0F1"

        # Parameters for waves
        width = 5.0
        x_max = 2 * PI
        
        # Helper to create sine wave
        def get_sine_wave(freq, color):
            return FunctionGraph(
                lambda x: 0.4 * np.sin(freq * x),
                x_range=[0, x_max],
                color=color
            ).stretch_to_fit_width(width)

        # === Animation for Lecture Line 1 ===
        # Real-world signals combine multiple sine waves together.
        self.lecture[0].set_color(YELLOW)
        
        red_wave = get_sine_wave(1, RED_C)
        green_wave = get_sine_wave(2.5, GREEN_C)
        blue_wave = get_sine_wave(4, BLUE_C)

        self.place_in_area(red_wave, "A1", "A6")
        self.place_in_area(green_wave, "B1", "B6")
        self.place_in_area(blue_wave, "C1", "C6")

        self.play(Create(red_wave), Create(green_wave), Create(blue_wave), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summing different frequencies creates a messy, complex wave.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # The sum wave
        sum_wave = FunctionGraph(
            lambda x: 0.3 * (np.sin(1 * x) + np.sin(2.5 * x) + np.sin(4 * x)),
            x_range=[0, x_max],
            color=WHITE_C
        ).stretch_to_fit_width(width)
        
        # [VideoCritic][section_3] Issue 51 fix:
        self.place_in_area(sum_wave, 'D1', 'E6')

        # Move the components to the sum wave position and transform
        target_center = sum_wave.get_center()

        self.play(
            red_wave.animate.move_to(target_center),
            green_wave.animate.move_to(target_center),
            blue_wave.animate.move_to(target_center),
            run_time=1.5
        )
        
        self.play(
            Transform(red_wave, sum_wave),
            Transform(green_wave, sum_wave),
            Transform(blue_wave, sum_wave),
            run_time=2
        )
        # Keep only one (now white) wave
        self.remove(green_wave, blue_wave)
        self.white_sum_wave = red_wave 
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This "Time Domain" view hides the individual components.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        td_label = Text("Time Domain", font_size=24, color=WHITE_C)
        
        # [VideoCritic][section_3] Issue 52 fix:
        self.place_in_area(td_label, 'F2', 'F5', scale_factor=0.8)

        # [AUTO-ASSET-INTEGRATION] Issue 44:
        signal_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/signal.svg")
        signal_icon.set_color(WHITE_C)
        self.place_at_grid(signal_icon, "F1", scale_factor=0.5)

        self.play(Write(td_label), FadeIn(signal_icon))
        self.wait(3)
