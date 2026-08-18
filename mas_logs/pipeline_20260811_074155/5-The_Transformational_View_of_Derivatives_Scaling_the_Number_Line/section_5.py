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
        # Setup layout
        title_text = "Application: The Elastic Chameleon"
        lecture_lines = [
            "Imagine skin pigment movement as a temperature function.",
            "High derivative means color shifts rapidly with heat.",
            "The derivative measures sensitivity to small temperature changes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for highlights
        color_1 = "#FFFF00"  # Yellow
        color_2 = "#FF4500"  # Orange-Red
        color_3 = "#00FFFF"  # Cyan

        # === Animation for Lecture Line 1 ===
        # Imagine skin pigment movement as a temperature function.
        self.lecture[0].set_color(color_1)
        
        # Temperature Line (Top)
        temp_line = NumberLine(
            x_range=[0, 10, 2],
            length=5,
            include_numbers=True,
            font_size=18,
            color=BLUE
        )
        self.place_in_area(temp_line, "B1", "B6")
        
        temp_label = Text("Temperature (°C)", font_size=20, color=BLUE)
        self.place_in_area(temp_label, "A2", "A5")

        # Chameleon Asset
        chameleon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png")
        self.place_at_grid(chameleon, "D1", scale_factor=0.6)

        # Skin Color Line (Bottom) with Gradient representing chameleon skin
        skin_line = NumberLine(
            x_range=[0, 10, 2],
            length=5,
            include_numbers=True,
            font_size=18,
            color=WHITE
        )
        self.place_in_area(skin_line, "E1", "E6")
        
        gradient_rect = Rectangle(
            width=5, height=0.3,
            fill_opacity=1.0,
            stroke_width=1,
            stroke_color=WHITE
        ).set_fill(color=[GREEN, YELLOW, RED, PURPLE])
        gradient_rect.move_to(skin_line.get_center())
        
        color_label = Text("Skin Pigment Position", font_size=20, color=WHITE)
        self.place_in_area(color_label, "D2", "D5")

        # Functional mapping: f(x) = 10 / (1 + exp(-4*(x-5)))
        # This function is very steep at x=5 (high sensitivity) and flat elsewhere.
        def f(x):
            return 10 / (1 + np.exp(-4 * (x - 5)))

        input_val = ValueTracker(2)
        
        # Slider on temperature line (input)
        temp_slider = Dot(color=BLUE, radius=0.1)
        temp_slider.add_updater(lambda d: d.move_to(temp_line.n2p(input_val.get_value())))
        
        # Indicator on skin color line (output)
        skin_indicator = Dot(color=color_2, radius=0.15)
        skin_indicator.add_updater(lambda d: d.move_to(skin_line.n2p(f(input_val.get_value()))))
        
        # Visual connection line (using updater instead of always_redraw)
        connect_line = Line(
            temp_slider.get_center(),
            skin_indicator.get_center(),
            color=GRAY,
            stroke_opacity=0.3
        )
        connect_line.add_updater(lambda l: l.set_points_as_corners([temp_slider.get_center(), skin_indicator.get_center()]))

        self.add(temp_line, temp_label, skin_line, gradient_rect, color_label, temp_slider, skin_indicator, connect_line, chameleon)
        self.play(
            FadeIn(temp_line), FadeIn(temp_label), 
            FadeIn(skin_line), FadeIn(gradient_rect), FadeIn(color_label),
            FadeIn(temp_slider), FadeIn(skin_indicator), FadeIn(connect_line),
            FadeIn(chameleon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # High derivative means color shifts rapidly with heat.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_2)
        
        high_sens_text = Text("High Sensitivity (High f')", font_size=24, color=color_2)
        self.place_in_area(high_sens_text, 'C1', 'C6', scale_factor=0.8)

        # Move to the high-sensitivity region near x=5
        self.play(input_val.animate.set_value(4.8), run_time=1.5)
        
        # Show rapid shift: a small input change results in a large output change
        self.play(
            input_val.animate.set_value(5.2),
            FadeIn(high_sens_text),
            run_time=3,
            rate_func=linear
        )
        self.play(Flash(high_sens_text, color=color_2, line_length=0.2), run_time=0.5)
        self.play(FadeOut(high_sens_text), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The derivative measures sensitivity to small temperature changes.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_3)

        low_sens_text = Text("Low Sensitivity (Low f')", font_size=24, color=color_3)
        self.place_in_area(low_sens_text, 'C1', 'C6', scale_factor=0.8)

        # Move to a low-sensitivity region (far from x=5)
        self.play(input_val.animate.set_value(7.5), run_time=1.5)
        
        # Show low sensitivity: a large input change results in almost no output change
        self.play(
            input_val.animate.set_value(9.5),
            FadeIn(low_sens_text),
            run_time=3,
            rate_func=linear
        )
        self.play(FadeOut(low_sens_text), run_time=0.5)
        self.wait(2)
