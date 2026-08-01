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
        # Initialize the layout with specific title and lecture lines
        self.setup_layout("Softmax: Turning Scores into Percentages", [
            "Softmax converts raw scores into clear probabilities.",
            "These weights always sum to one hundred percent.",
            "They determine how much each word contributes."
        ])

        # Color palette for consistency
        color1 = YELLOW
        color2 = GREEN
        color3 = TEAL

        # === Animation for Lecture Line 1 ===
        # Goal: Display raw scores (10, 2, 1) as a bar chart.
        self.play(self.lecture[0].animate.set_color(color1))

        # Raw scores data
        raw_vals = [10, 2, 1]
        h_scale = 0.2
        
        bar1 = Rectangle(width=0.6, height=raw_vals[0]*h_scale, fill_opacity=0.8, fill_color=color1, stroke_color=WHITE)
        bar2 = Rectangle(width=0.6, height=raw_vals[1]*h_scale, fill_opacity=0.8, fill_color=color1, stroke_color=WHITE)
        bar3 = Rectangle(width=0.6, height=raw_vals[2]*h_scale, fill_opacity=0.8, fill_color=color1, stroke_color=WHITE)
        
        raw_bars = VGroup(bar1, bar2, bar3).arrange(RIGHT, aligned_edge=DOWN, buff=0.4)
        # Centering the chart in the right side area
        self.place_in_area(raw_bars, "B2", "D5")

        label1 = Text("10", font_size=20).next_to(bar1, UP, buff=0.1)
        label2 = Text("2", font_size=20).next_to(bar2, UP, buff=0.1)
        label3 = Text("1", font_size=20).next_to(bar3, UP, buff=0.1)
        labels = VGroup(label1, label2, label3)
        
        self.play(Create(raw_bars), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Goal: Transform bars to percentages (80%, 15%, 5%) and show sum.
        self.play(self.lecture[1].animate.set_color(color2))

        # Probability heights (scaled)
        prob_h = [2.0, 0.375, 0.125]
        
        new_bar1 = Rectangle(width=0.6, height=prob_h[0], fill_opacity=0.8, fill_color=color2, stroke_color=WHITE)
        new_bar2 = Rectangle(width=0.6, height=prob_h[1], fill_opacity=0.8, fill_color=color2, stroke_color=WHITE)
        new_bar3 = Rectangle(width=0.6, height=prob_h[2], fill_opacity=0.8, fill_color=color2, stroke_color=WHITE)
        
        prob_bars = VGroup(new_bar1, new_bar2, new_bar3).arrange(RIGHT, aligned_edge=DOWN, buff=0.4)
        # Position using the same area to ensure alignment
        self.place_in_area(prob_bars, "B2", "D5")
        
        new_label1 = Text("80%", font_size=20, color=color2).next_to(new_bar1, UP, buff=0.1)
        new_label2 = Text("15%", font_size=20, color=color2).next_to(new_bar2, UP, buff=0.1)
        new_label3 = Text("5%", font_size=20, color=color2).next_to(new_bar3, UP, buff=0.1)
        new_labels = VGroup(new_label1, new_label2, new_label3)

        # Summation label - Fixed position according to Issue 34
        sum_label = Text("Sum = 100%", font_size=24, color="#00FF00")
        self.place_at_grid(sum_label, "D3") # Addressing Issue 34

        self.play(
            Transform(raw_bars, prob_bars),
            Transform(labels, new_labels),
            run_time=1.5
        )
        self.play(Write(sum_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Goal: Sentence spotlight on 'River' in 'River bank'.
        self.play(self.lecture[2].animate.set_color(color3))

        # Cleanup bar chart
        self.play(FadeOut(raw_bars), FadeOut(labels), FadeOut(sum_label))

        # Setup word boxes for 'River bank'
        # Smaller width to avoid overlap at adjacent grid points
        box_w = 0.9
        river_box = RoundedRectangle(corner_radius=0.1, width=box_w, height=0.8, color=WHITE)
        river_text = Text("River", font_size=20)
        
        # Load asset
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg"
        try:
            river_icon = SVGMobject(asset_path).scale(0.3)
            river_icon.next_to(river_box, DOWN, buff=0.1)
            river_word = VGroup(river_box, river_text, river_icon)
        except Exception:
            river_word = VGroup(river_box, river_text)
        
        bank_box = RoundedRectangle(corner_radius=0.1, width=box_w, height=0.8, color=WHITE)
        bank_text = Text("bank", font_size=20)
        bank_word = VGroup(bank_box, bank_text)

        # Positioning based on Issue 32 & 33
        self.place_at_grid(river_word, "C3") # Addressing Issue 32
        self.place_at_grid(bank_word, "C4")  # Addressing Issue 33

        self.play(FadeIn(river_word), FadeIn(bank_word))
        self.wait(0.5)

        # Highlight 'River' with a spotlight circle and 80% weight
        spotlight = Circle(radius=0.8, color=color2, fill_opacity=0.2, stroke_width=2).move_to(river_word)
        label_80 = Text("80%", font_size=20, color=color2).next_to(river_word, UP, buff=0.2)
        
        self.play(
            river_box.animate.set_stroke(color=color2, width=5),
            bank_word.animate.set_opacity(0.3),
            Create(spotlight),
            Write(label_80)
        )
        self.wait(2)
