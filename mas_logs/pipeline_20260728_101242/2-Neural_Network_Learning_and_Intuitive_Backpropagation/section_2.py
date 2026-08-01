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
        # Define lecture lines
        lecture_lines = [
            "- Neurons process inputs through weighted connections.",
            "- Weights determine the significance of each input.",
            "- Adjusting these weights changes the final output signal."
        ]
        
        self.setup_layout("Prerequisite: The Weighted Connection", lecture_lines)
        
        # Colors
        COLOR_NEURON = "#00FFFF"
        COLOR_WEIGHT = "#FFFF00"
        COLOR_FLAVOR = "#FF00FF"
        
        # Trackers
        weight_tracker = ValueTracker(0.5)
        
        # === Animation for Lecture Line 1 ===
        # "Neurons process inputs through weighted connections."
        # Neuron circle (#00FFFF) processes 'Salt' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/salt.svg]
        # input through a connection with a yellow weight (#FFFF00).
        
        self.play(self.lecture[0].animate.set_color(COLOR_NEURON))
        
        # Assets
        salt_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/salt.svg", color=WHITE)
        self.place_at_grid(salt_icon, 'C1', scale_factor=0.6)
        salt_label = Text("Salt", font_size=18, color=WHITE).next_to(salt_icon, DOWN, buff=0.1)
        
        neuron = Circle(radius=0.5, color=COLOR_NEURON, fill_opacity=0.2)
        neuron_label = Text("Neuron", font_size=18, color=COLOR_NEURON)
        neuron_label.next_to(neuron, UP, buff=0.2)
        neuron_group = VGroup(neuron, neuron_label)
        # Fix Issue 34: self.place_in_area(neuron_group, 'B3', 'C3', scale_factor=0.9)
        self.place_in_area(neuron_group, 'B3', 'C3', scale_factor=0.9)
        
        # Connection line/arrow
        input_arrow = Arrow(self.grid['C1'], neuron.get_left(), buff=0.1, color=WHITE)
        
        weight_label = Text("Weight", font_size=18, color=COLOR_WEIGHT)
        weight_label.next_to(input_arrow, UP, buff=0.1)
        
        self.play(
            FadeIn(salt_icon),
            Write(salt_label),
            Create(neuron_group),
            GrowArrow(input_arrow),
            Write(weight_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # "Weights determine the significance of each input."
        # Show 'Flavor Score' (#FF00FF) impact, highlighting the weight significance on the connection line.
        
        self.play(self.lecture[1].animate.set_color(COLOR_FLAVOR))
        
        # Fix Issue 33: self.place_in_area(flavor_label, 'C5', 'C6', scale_factor=0.8)
        flavor_label = Text("Flavor Score", font_size=20, color=COLOR_FLAVOR)
        self.place_in_area(flavor_label, 'C5', 'C6', scale_factor=0.8)
        flavor_label.shift(UP * 0.4) # Slight lift within the area to make room for value
        
        output_arrow = Arrow(neuron.get_right(), flavor_label.get_left(), buff=0.2, color=WHITE)
        
        score_value = DecimalNumber(weight_tracker.get_value() * 10, num_decimal_places=1, color=COLOR_FLAVOR)
        score_value.next_to(flavor_label, DOWN, buff=0.2)
        score_value.add_updater(lambda d: d.set_value(weight_tracker.get_value() * 10))
        
        # Highlighting connection line impact
        highlight_circle = Circle(radius=0.4, color=COLOR_WEIGHT).move_to(weight_label.get_center())
        
        self.play(
            GrowArrow(output_arrow),
            Write(flavor_label),
            Write(score_value),
            Create(highlight_circle)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # "Adjusting these weights changes the final output signal."
        # Moving the Weight slider (#FFFF00) changes the final Flavor Score output signal.
        
        self.play(self.lecture[2].animate.set_color(COLOR_WEIGHT))
        
        # Slider elements
        slider_line = Line(self.grid['E2'], self.grid['E4'], color=WHITE)
        slider_knob = Dot(color=COLOR_WEIGHT)
        
        # Positioning the knob based on ValueTracker (mapping 0.0-1.0 to line)
        slider_knob.add_updater(lambda m: m.move_to(slider_line.point_from_proportion(weight_tracker.get_value())))
        
        slider_text = Text("Weight Slider", font_size=18, color=COLOR_WEIGHT)
        slider_text.next_to(slider_line, UP, buff=0.2)
        
        self.play(
            Create(slider_line),
            FadeIn(slider_knob),
            Write(slider_text)
        )
        self.wait(0.5)
        
        # Animate weight change and observe Flavor Score update
        self.play(weight_tracker.animate.set_value(0.9), run_time=1.5)
        self.wait(0.5)
        self.play(weight_tracker.animate.set_value(0.2), run_time=1.5)
        self.wait(0.5)
        
        # Final connection highlight
        self.play(
            Indicate(weight_label, color=COLOR_WEIGHT, scale_factor=1.2),
            Indicate(input_arrow, color=COLOR_WEIGHT),
            weight_tracker.animate.set_value(0.7)
        )
        self.wait(2)
