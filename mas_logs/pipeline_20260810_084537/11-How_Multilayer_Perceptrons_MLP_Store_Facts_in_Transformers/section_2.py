from manim import *

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
        self.setup_layout("Prerequisite: The Neuron as a Pattern Matcher", [
            "Neurons function as pattern-matching light dimmers.",
            "Input matches weight pattern; neuron fires.",
            "Higher layers recognize more abstract concepts."
        ])
        
        # Assets
        neuron_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg", color=WHITE)
        dimmer_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dimmer.svg")
        
        neuron_label = Text("Neuron", font_size=18)
        neuron_group = VGroup(neuron_svg, neuron_label).arrange(DOWN)
        # Applied fix for #29 and #44
        self.place_at_grid(neuron_group, "C3", scale_factor=0.8)

        input_signal = Dot(color=ORANGE)
        output_signal = Dot(color=YELLOW)
        weight_conn = Line(start=ORIGIN, end=RIGHT*1.5, color=GREEN, stroke_width=4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(neuron_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        # Applied fix for #30 and #45
        self.place_in_area(weight_conn, "C1", "C5", scale_factor=0.7)
        self.play(FadeIn(input_signal.move_to(self.grid["C1"])), Create(weight_conn))
        
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(output_signal, "C5")
        self.play(
            input_signal.animate.move_to(neuron_group.get_center()),
            FadeIn(output_signal),
            neuron_group.animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        pattern_text = Text("Pattern Matched", font_size=24, color="#00FFFF")
        # Applied fix for #31 and #46 and integration of dimmer asset
        self.place_at_grid(pattern_text, "E5", scale_factor=0.9)
        self.place_at_grid(dimmer_svg, "E2", scale_factor=0.5)
        self.play(Write(pattern_text), FadeIn(dimmer_svg))
        self.wait(2)
