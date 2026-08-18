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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Weights & Biases", [
            "Neurons use \"knobs\" called weights to process data.",
            "Each knob determines an input's influence on results.",
            "A bias sets the starting point for every decision."
        ])

        # === Animation for Lecture Line 1 ===
        # Display labels 'Temperature' (#FFD700) and 'Size' (#ADFF2F) with numerical inputs.
        # Resolving Issue 29 and 30 regarding spacing and overlaps.
        self.lecture[0].set_color("#FFD700")
        
        temp_label = Text("Temperature", font_size=24, color="#FFD700")
        temp_value = Text("72°", font_size=32, color="#FFD700")
        self.place_in_area(temp_label, "B2", "B3", scale_factor=0.8)
        self.place_at_grid(temp_value, "B4", scale_factor=0.8)
        
        size_label = Text("Size", font_size=24, color="#ADFF2F")
        size_value = Text("10", font_size=32, color="#ADFF2F")
        self.place_in_area(size_label, "D2", "D3", scale_factor=0.8)
        self.place_at_grid(size_value, "D4", scale_factor=0.8)
        
        self.play(
            FadeIn(temp_label), FadeIn(temp_value),
            FadeIn(size_label), FadeIn(size_value),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show two dial icons labeled 'Weights' (#00BFFF) that rotate to show adjustment.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00BFFF")
        
        weight_color = "#00BFFF"
        
        # Dial 1 (Connected to Temperature)
        dial1_circle = Circle(radius=0.4, color=weight_color)
        dial1_pointer = Line(dial1_circle.get_center(), dial1_circle.get_top(), color=weight_color)
        dial1 = VGroup(dial1_circle, dial1_pointer)
        self.place_at_grid(dial1, "B5")
        
        # Dial 2 (Connected to Size)
        dial2_circle = Circle(radius=0.4, color=weight_color)
        dial2_pointer = Line(dial2_circle.get_center(), dial2_circle.get_top(), color=weight_color)
        dial2 = VGroup(dial2_circle, dial2_pointer)
        self.place_at_grid(dial2, "D5")
        
        weights_label = Text("Weights", font_size=24, color=weight_color)
        self.place_at_grid(weights_label, "C5")
        
        # Arrows indicating flow from values to weights
        arrow1 = Arrow(temp_value.get_right(), dial1.get_left(), buff=0.1, color=WHITE)
        arrow2 = Arrow(size_value.get_right(), dial2.get_left(), buff=0.1, color=WHITE)
        
        self.play(
            Create(dial1_circle), Create(dial2_circle),
            Create(dial1_pointer), Create(dial2_pointer),
            Write(weights_label),
            GrowArrow(arrow1), GrowArrow(arrow2),
            run_time=1
        )
        
        self.play(
            Rotate(dial1_pointer, angle=-PI/2, about_point=dial1_circle.get_center()),
            Rotate(dial2_pointer, angle=PI/3, about_point=dial2_circle.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a 'Bias' slider (#FF69B4) moving horizontally to set the starting point.
        # Resolving Issue 31: Aligning bias label with the slider.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF69B4")
        
        bias_color = "#FF69B4"
        
        slider_line = Line(LEFT, RIGHT, color=bias_color).scale(1.5)
        slider_knob = Dot(color=bias_color).move_to(slider_line.get_left())
        bias_slider = VGroup(slider_line, slider_knob)
        self.place_in_area(bias_slider, "F2", "F5")
        
        bias_label = Text("Bias", font_size=24, color=bias_color)
        self.place_in_area(bias_label, "E3", "E4", scale_factor=0.8)
        
        self.play(
            Create(slider_line),
            FadeIn(slider_knob),
            Write(bias_label),
            run_time=1
        )
        
        self.play(
            slider_knob.animate.move_to(slider_line.get_center()),
            run_time=1.5
        )
        self.play(
            slider_knob.animate.move_to(slider_line.get_right()),
            run_time=1.5
        )
        self.wait(2)
