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
        self.setup_layout("The Core Innovation: Self-Attention Mechanism", [
            "Transformers analyze entire sentences simultaneously.",
            "Self-attention calculates relevance scores between words.",
            "Context depends on other sentence words.",
            "'It' links to 'animal' in context.",
            "Final attention weights visualization."
        ])
        
        words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
        word_mobs = VGroup(*[Text(w, font_size=20) for w in words]).arrange(RIGHT, buff=0.2)
        
        # Use SVGMobject for animal icon
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(word_mobs, "A1", "A6", scale_factor=0.9)
        self.place_at_grid(animal_icon.copy(), "B3", scale_factor=0.3)
        self.play(FadeIn(word_mobs), FadeIn(animal_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        it_idx = 7
        animal_idx = 1
        line = Line(word_mobs[it_idx].get_bottom(), word_mobs[animal_idx].get_bottom(), color=YELLOW)
        self.play(Create(line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        rect = SurroundingRectangle(word_mobs, color=GREEN, buff=0.1)
        self.play(Create(rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        attention_dot = Dot(color="#FF00FF")
        self.place_at_grid(attention_dot, "C3", scale_factor=0.8)
        self.play(FadeIn(attention_dot))
        self.play(attention_dot.animate.move_to(word_mobs[it_idx].get_top() + UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        final_animal = animal_icon.copy()
        self.place_at_grid(final_animal, "E3", scale_factor=0.5)
        self.play(FadeIn(final_animal))
        self.wait(1)
