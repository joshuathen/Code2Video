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
        title = "The Core Intuition: The 'It' Problem"
        lines = [
            "Language depends heavily on surrounding context.",
            "Consider the ambiguous word \"it\" in sentences.",
            "Attention helps the computer focus on relevant words."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow for 'it'
        COLOR_ANIMAL = "#00FF00"    # Green
        COLOR_STREET = "#0000FF"    # Blue
        COLOR_WHITE = "#FFFFFF"
        COLOR_GRAY = "#888888"

        # === Animation for Lecture Line 1 ===
        # Animation: Display the sentence 'The animal didn't cross the street because it was too tired.' in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        s1_words_list = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired."]
        s1 = VGroup(*[Text(word, font_size=20, color=COLOR_WHITE) for word in s1_words_list]).arrange(RIGHT, buff=0.1)
        self.place_in_area(s1, 'B1', 'B6', scale_factor=1.0)
        
        self.play(Write(s1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animation: Highlight 'it' in #FFFF00 and 'animal' in #00FF00, then draw a #00FF00 arrow from 'it' to 'animal'.
        self.play(
            self.lecture[0].animate.set_color(COLOR_GRAY),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Indices: animal(1), it(7)
        it_word_1 = s1[7]
        animal_word_1 = s1[1]
        
        self.play(
            it_word_1.animate.set_color(COLOR_HIGHLIGHT),
            animal_word_1.animate.set_color(COLOR_ANIMAL)
        )
        
        # Use CurvedArrow for better visual connection
        arrow1 = CurvedArrow(it_word_1.get_top(), animal_word_1.get_top(), color=COLOR_ANIMAL, angle=-TAU/4)
        self.play(Create(arrow1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animation: Transition to second sentence and highlight new relationships.
        self.play(
            self.lecture[1].animate.set_color(COLOR_GRAY),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        self.play(FadeOut(s1), FadeOut(arrow1))
        
        s2_words_list = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "wide."]
        s2 = VGroup(*[Text(word, font_size=20, color=COLOR_WHITE) for word in s2_words_list]).arrange(RIGHT, buff=0.1)
        self.place_in_area(s2, 'B1', 'B6', scale_factor=1.0)
        
        self.play(Write(s2))
        
        # Indices: street(5), it(7)
        it_word_2 = s2[7]
        street_word_2 = s2[5]
        
        self.play(
            it_word_2.animate.set_color(COLOR_HIGHLIGHT),
            street_word_2.animate.set_color(COLOR_STREET)
        )
        
        arrow2 = CurvedArrow(it_word_2.get_bottom(), street_word_2.get_bottom(), color=COLOR_STREET, angle=TAU/4)
        self.play(Create(arrow2))
        
        # Flash labels for 'animal' and 'street' to emphasize reference shift.
        label_animal = Text("REF: ANIMAL", font_size=18, color=COLOR_ANIMAL)
        label_street = Text("REF: STREET", font_size=18, color=COLOR_STREET)
        
        # Positioning at Row D to avoid overlap with indicators (Fixes Issue 24 & 25)
        self.place_at_grid(label_animal, 'D2', scale_factor=0.8)
        self.place_at_grid(label_street, 'D5', scale_factor=0.8)
        
        self.play(
            FadeIn(label_animal),
            FadeIn(label_street),
            Flash(label_animal, color=COLOR_ANIMAL),
            Flash(label_street, color=COLOR_STREET)
        )
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(COLOR_WHITE))
        self.wait(1)
