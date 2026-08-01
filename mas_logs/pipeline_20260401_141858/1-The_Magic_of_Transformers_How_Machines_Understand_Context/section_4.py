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

class Section4Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "The Core Mechanism: Self-Attention"
        lecture_lines = [
            "Transformers analyze every word in a sentence simultaneously.",
            "Self-attention determines which words relate to each other.",
            "It resolves ambiguity by focusing on relevant context.",
            "'It' connects strongly to 'animal' using attention weights.",
            "This uses the math of Query and Key vectors."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define the sentence words and grid positions
        # Sentence: "The animal didn't cross the street because it was too tired."
        words_list = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."]
        word_objs = VGroup(*[Text(w, font_size=24, color=WHITE) for w in words_list])
        
        # Mapping words to B and C rows (6 words per row)
        grid_positions = [
            "B1", "B2", "B3", "B4", "B5", "B6",
            "C1", "C2", "C3", "C4", "C5", "C6"
        ]
        
        for obj, pos in zip(word_objs, grid_positions):
            self.place_at_grid(obj, pos)

        # Reference specific words for easier access
        animal_word = word_objs[1]
        street_word = word_objs[5]
        it_word = word_objs[7]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(word_objs))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight 'it' (#FFFF00) and make it pulse
        self.play(it_word.animate.set_color("#FFFF00"))
        self.play(Indicate(it_word, color="#FFFF00", scale_factor=1.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Draw connecting lines from 'it' to every other word
        attention_lines = VGroup()
        for i, word in enumerate(word_objs):
            if i == 7: continue # skip 'it' itself
            line = Line(
                it_word.get_center(), 
                word.get_center(), 
                stroke_width=1, 
                stroke_opacity=0.4, 
                color=WHITE
            )
            attention_lines.add(line)
        
        self.play(Create(attention_lines))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Identify the lines to 'animal' (index 1) and 'street' (index 5)
        line_to_animal = attention_lines[1]
        line_to_street = attention_lines[5]
        
        self.play(
            line_to_animal.animate.set_stroke(color="#00FF00", width=8, opacity=1),
            line_to_street.animate.set_stroke(color="#FF0000", width=1, opacity=0.3),
            animal_word.animate.set_color("#00FF00"),
            street_word.animate.set_color("#FF0000")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Display 'Attention Weights' as a heat-map bar (#FFFFFF) below the sentence
        heatmap_label = Text("Attention Weights", font_size=18, color=WHITE)
        # Fix for Issue 43: Area E2-E4, scale 0.7
        self.place_in_area(heatmap_label, "E2", "E4", scale_factor=0.7)
        
        heatmap_bar = Rectangle(height=0.4, width=4.0, color=WHITE, stroke_width=2)
        # Fix for Issue 44: Area F2-F6
        self.place_in_area(heatmap_bar, "F2", "F6")
        
        # Visual representations aligned with animal (Col 2) and street (Col 6)
        focus_animal = Rectangle(height=0.3, width=0.7, color="#00FF00", fill_opacity=0.8, stroke_width=0)
        self.place_at_grid(focus_animal, "F2")
        
        focus_street = Rectangle(height=0.3, width=0.7, color="#FF0000", fill_opacity=0.2, stroke_width=0)
        self.place_at_grid(focus_street, "F6")
        
        # Q and K math labels
        q_label = Text("Query (Q)", font_size=14, color="#FFFF00")
        # Fix for Issue 42: D2, scale 0.8
        self.place_at_grid(q_label, "D2", scale_factor=0.8)
        
        k_label = Text("Keys (K)", font_size=14, color=WHITE)
        self.place_at_grid(k_label, "A2", scale_factor=0.8)

        self.play(
            FadeIn(heatmap_label),
            FadeIn(heatmap_bar),
            FadeIn(focus_animal),
            FadeIn(focus_street),
            FadeIn(q_label),
            FadeIn(k_label)
        )
        self.wait(2)
        
        # Final state
        self.lecture[4].set_color(WHITE)
        self.wait(2)
