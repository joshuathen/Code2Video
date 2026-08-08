from manim import *
import os

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
        lecture_lines = [
            "Traditional models process words in isolation.",
            "Human language relies on context and focus.",
            "\"It\" refers back to \"animal\" in this sentence."
        ]
        self.setup_layout("The Intuition: Contextual Focus", lecture_lines)
        
        sentence = ["The", "animal", "didn't", "cross", "it"]
        words = VGroup(*[Text(w, font_size=28) for w in sentence]).arrange(RIGHT, buff=0.3)
        self.place_in_area(words, 'A4', 'B6', scale_factor=0.6)

        # Load Animal Asset
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        animal_icon.scale(0.2).next_to(words[1], UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(words), FadeIn(animal_icon))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF") # Highlight color
        highlight = SurroundingRectangle(words[4], color="#FF00FF", buff=0.1)
        self.play(Create(highlight))
        
        context_window = Rectangle(width=1.5, height=0.8, color="#00FFFF", fill_opacity=0.2)
        self.place_at_grid(context_window, 'E5', scale_factor=0.7)
        self.play(FadeIn(context_window))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00") # Yellow
        line = Line(words[4].get_center(), words[1].get_center(), color="#FFFF00")
        self.play(Create(line))
        
        # Interaction group for C2-D3
        interaction_elements = VGroup(highlight, context_window, line)
        self.place_in_area(interaction_elements, 'C2', 'D3', scale_factor=0.85)

        self.play(
            FadeOut(highlight),
            FadeOut(context_window),
            FadeOut(words[0]),
            FadeOut(words[2]),
            FadeOut(words[3])
        )
