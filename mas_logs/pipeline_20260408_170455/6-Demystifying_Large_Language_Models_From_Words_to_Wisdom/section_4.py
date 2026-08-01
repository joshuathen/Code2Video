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
        self.setup_layout(
            "Self-Attention: The Context Spotlight", 
            [
                'Self-attention spotlights the most relevant words in a sentence.', 
                'It resolves ambiguity by looking at the surrounding context.', 
                'When reading "it", the model highlights the subject "animal".', 
                'Different contexts shift the spotlight to different words.', 
                'This process creates a weighted understanding of every word.'
            ]
        )

        # Helper to create the sentence word by word for easy manipulation
        def create_sentence(words_list):
            vg = VGroup()
            for w in words_list:
                vg.add(Text(w, font_size=24, color=WHITE))
            vg.arrange(RIGHT, buff=0.2)
            return vg

        sentence_words_1 = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired."]
        sentence_vgroup = create_sentence(sentence_words_1)
        
        # FIX: Move sentence_vgroup to avoid overlap (Issue 49, 62)
        self.place_in_area(sentence_vgroup, "B1", "B6", scale_factor=0.8)

        # Asset Integration (Issue 38, 62)
        animal_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/animal.svg")
        self.place_at_grid(animal_icon, "A2", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(sentence_vgroup), FadeIn(animal_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Word 'it' is index 7
        it_word = sentence_vgroup[7]
        self.play(
            it_word.animate.set_color("#FFFF00").scale(1.2),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Spotlight from 'it' to 'animal' (index 1)
        animal_word = sentence_vgroup[1]
        
        def get_spotlight(source, target):
            points = [
                source.get_top(),
                target.get_top() + UP * 0.1,
                target.get_bottom() + DOWN * 0.1,
                source.get_bottom()
            ]
            return Polygon(*points, color="#FFFF00", fill_opacity=0.3, stroke_width=0)

        spotlight = get_spotlight(it_word, animal_word)
        self.play(FadeIn(spotlight))
        self.play(animal_word.animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Transform 'tired' to 'wide'
        tired_word = sentence_vgroup[10]
        # Align 'wide.' with the center of 'tired.'
        wide_word = Text("wide.", font_size=24, color=WHITE).scale(0.8).move_to(tired_word.get_center())
        
        # New spotlight target 'street' (index 5)
        street_word = sentence_vgroup[5]
        new_spotlight = get_spotlight(it_word, street_word)

        self.play(
            Transform(tired_word, wide_word),
            Transform(spotlight, new_spotlight),
            animal_word.animate.set_color(WHITE),
            street_word.animate.set_color("#FFFF00"),
            animal_icon.animate.set_opacity(0.2), # Fade out animal icon as context shifts
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Create Heatmap grid for 'it'
        heatmap = VGroup()
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 1.0, 0.1, 0.1, 0.1] # 'it' and 'street' are high
        
        for i, weight in enumerate(weights):
            sq = Square(side_length=0.3, fill_color=YELLOW, fill_opacity=weight, stroke_color=WHITE, stroke_width=1)
            sq.move_to(sentence_vgroup[i].get_center() + DOWN * 0.8)
            heatmap.add(sq)

        heatmap_label = Text("Attention Weights", font_size=16, color=WHITE)
        # FIX: Move heatmap_label to avoid obstruction (Issue 50, 62)
        self.place_at_grid(heatmap_label, "D1", scale_factor=0.8)
        # Re-align label relative to heatmap (using grid constraints as guideline)
        heatmap_label.next_to(heatmap, LEFT, buff=0.3)

        self.play(FadeIn(heatmap), Write(heatmap_label))
        self.wait(2)

        # Finish
        self.lecture[4].set_color(WHITE)
        self.wait(2)
