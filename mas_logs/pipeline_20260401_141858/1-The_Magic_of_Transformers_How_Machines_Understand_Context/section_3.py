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
        self.setup_layout(
            "The Problem: The 'Bank' Dilemma", 
            [
                "Old models process words sequentially and forget context.", 
                "This creates confusion for words with multiple meanings.", 
                "Does 'bank' mean a river edge or money storage?"
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Sentence 1: "I went to the bank to deposit money."
        s1_text_list = ["I", "went", "to", "the", "bank", "to", "deposit", "money."]
        s1_words = VGroup(*[Text(w, font_size=24) for w in s1_text_list])
        s1_words.arrange(RIGHT, buff=0.15)
        # Resolved Issue 40: Adjusted area and scale to prevent cramping
        self.place_in_area(s1_words, 'B1', 'B5', scale_factor=0.9)
        
        # Sentence 2: "I sat on the river bank."
        s2_text_list = ["I", "sat", "on", "the", "river", "bank."]
        s2_words = VGroup(*[Text(w, font_size=24) for w in s2_text_list])
        s2_words.arrange(RIGHT, buff=0.15)
        # Resolved Issue 41: Adjusted area and scale for balance
        self.place_in_area(s2_words, 'D1', 'D5', scale_factor=0.9)

        # Sequence simulation: words appear one by one to show "sequential" processing
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(
            LaggedStart(*[Write(w) for w in s1_words], lag_ratio=0.1),
            LaggedStart(*[Write(w) for w in s2_words], lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Resolved Issue 32: [AUTO-ASSET-INTEGRATION]
        # Load and place referenced icon assets
        money_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/money.svg", fill_color=WHITE)
        river_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/river.svg", fill_color=WHITE)
        
        # Place assets near the relevant words
        self.place_at_grid(money_icon, 'A5', scale_factor=0.3)
        self.place_at_grid(river_icon, 'E4', scale_factor=0.3)
        
        # Draw connecting arrows from assets to ambiguous 'bank' word
        # s1_words[4] is "bank", s2_words[5] is "bank."
        arrow1 = CurvedArrow(
            start_point=money_icon.get_left(), 
            end_point=s1_words[4].get_top() + UP*0.1, 
            angle=PI/4, 
            color=WHITE
        )
        arrow2 = CurvedArrow(
            start_point=river_icon.get_right(), 
            end_point=s2_words[5].get_bottom() + DOWN*0.1, 
            angle=-PI/4, 
            color=WHITE
        )

        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(
            FadeIn(money_icon), FadeIn(river_icon),
            Create(arrow1), Create(arrow2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Change bank colors: #00FF00 (money context) and #3399FF (river context)
        # We also color icons and related words to emphasize the context-meaning link
        
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(
            s1_words[4].animate.set_color("#00FF00"), # "bank" in money context
            s2_words[5].animate.set_color("#3399FF"), # "bank" in river context
            s1_words[7].animate.set_color("#00FF00"), # "money."
            s2_words[4].animate.set_color("#3399FF"), # "river"
            money_icon.animate.set_color("#00FF00"),
            river_icon.animate.set_color("#3399FF"),
            arrow1.animate.set_color("#00FF00"),
            arrow2.animate.set_color("#3399FF")
        )
        self.wait(2)
