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
        # Data from shared state
        title_str = "The Bridge: From Discrete to Continuous"
        lecture_lines = [
            "Discrete variables represent countable outcomes like die rolls.",
            "Continuous variables like a kitten's weight have infinite possibilities.",
            "We can divide discrete bars into many narrow columns.",
            "These columns eventually smooth into a continuous curve.",
            "This curve is the bridge to understanding continuous data."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors from storyboard
        BLUE_B = "#3498db"
        KITTEN_C = "#ecf0f1"
        RED_C = "#e74c3c"

        # === Animation for Lecture Line 1 ===
        # Show blue discrete bars representing a die roll.
        self.lecture[0].set_color(BLUE_B)
        
        bar_heights = [1.2, 2.0, 1.5, 2.4, 1.8, 1.0]
        bars = VGroup(*[
            Rectangle(
                width=0.4, height=h, 
                fill_opacity=0.8, fill_color=BLUE_B, 
                stroke_color=WHITE, stroke_width=1
            )
            for h in bar_heights
        ]).arrange(RIGHT, buff=0.2)
        
        bar_labels = VGroup(*[
            Text(str(i+1), font_size=18, color=WHITE) for i in range(len(bar_heights))
        ])
        for i, label in enumerate(bar_labels):
            label.next_to(bars[i], DOWN, buff=0.1)
            
        discrete_group = VGroup(bars, bar_labels)
        self.place_in_area(discrete_group, "C1", "E6", scale_factor=0.9)
        
        self.play(FadeIn(discrete_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Display a kitten icon (#ecf0f1) with changing weight.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/kitten.svg]
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(KITTEN_C)
        
        kitten_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/kitten.svg")
        kitten_icon.set_color(KITTEN_C)
        self.place_at_grid(kitten_icon, "A2", scale_factor=0.6)
        
        # Changing weight label
        weight_tracker = ValueTracker(1.245)
        # We build DecimalNumber once, update in place
        weight_label = DecimalNumber(1.245, num_decimal_places=3, include_sign=False, unit=" kg", font_size=24, color=KITTEN_C)
        weight_label.add_updater(lambda d: d.set_value(weight_tracker.get_value()))
        # Fix Issue 27: weight_label to A3 (within 1 unit of A2)
        self.place_at_grid(weight_label, "A3")
        
        self.play(FadeIn(kitten_icon), FadeIn(weight_label))
        self.play(weight_tracker.animate.set_value(1.350), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transform the bars into many thin, narrow columns.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE_B)
        
        thin_bars = VGroup(*[
            Rectangle(
                width=0.08, height=1.5 + 0.5 * np.sin(x),
                fill_opacity=0.6, fill_color=BLUE_B, stroke_width=0
            )
            for x in np.linspace(0, 4*PI, 40)
        ]).arrange(RIGHT, buff=0.02)
        self.place_in_area(thin_bars, "C1", "E6", scale_factor=0.9)
        
        pet_label = Text("Digital Pet: 1 or 2 kg", font_size=20, color=BLUE_B)
        # Fix Issue 28: place_in_area for 5-word label
        self.place_in_area(pet_label, "B1", "B3")
        
        self.play(
            ReplacementTransform(bars, thin_bars),
            FadeOut(bar_labels),
            Write(pet_label)
        )
        self.play(weight_tracker.animate.set_value(2.000), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Smooth the columns into a continuous red curve (#e74c3c).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(KITTEN_C)
        
        curve = FunctionGraph(
            lambda x: 1.2 * np.exp(-x**2/2) + 0.5,
            x_range=[-2.5, 2.5],
            color=RED_C,
            stroke_width=4
        )
        self.place_in_area(curve, "C1", "E6", scale_factor=0.9)
        
        kitten_label = Text("Real Kitten: Continuous", font_size=20, color=KITTEN_C)
        self.place_at_grid(kitten_label, "B5")
        
        self.play(
            ReplacementTransform(thin_bars, curve),
            FadeIn(kitten_label),
            FadeOut(pet_label),
            weight_tracker.animate.set_value(1.567),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the curve as the 'Bridge' to continuity.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(RED_C)
        
        bridge_label = Text("The Bridge to Continuity", font_size=24, color=RED_C)
        # Fix Issue 29: place_in_area for 4-word label
        self.place_in_area(bridge_label, "F3", "F5")
        
        self.play(
            curve.animate.set_stroke(width=8),
            Write(bridge_label)
        )
        self.play(Indicate(curve, color=YELLOW))
        self.wait(2)
