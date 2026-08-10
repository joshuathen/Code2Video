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
        lecture_lines = ["Attention provides a dynamic spotlight mechanism.", "Highly relevant words cast a brighter glow.", "Example: 'it' focuses its light on 'cat'."]
        self.setup_layout("Core Concept: The 'Spotlight' Analogy", lecture_lines)
        
        # Text setup
        sentence = Text("The cat sat on the mat because it was tired", font_size=24)
        self.place_in_area(sentence, 'B2', 'B5', scale_factor=0.75)
        
        # Asset setup
        document_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg")
        self.place_at_grid(document_icon, 'A2', scale_factor=0.5)
        
        # Spotlight setup
        spotlight = Circle(radius=0.5, color=YELLOW, fill_opacity=0.3)
        spotlight.move_to(self.grid['C3'])
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sentence), FadeIn(document_icon), run_time=1)
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(spotlight))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(spotlight.animate.shift(RIGHT * 1.5), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Using slice indexing for Text objects in Manim CE
        it_pos = sentence[35:37].get_center()
        cat_pos = sentence[4:7].get_center()
        
        self.play(spotlight.animate.move_to(it_pos))
        self.play(spotlight.animate.move_to(cat_pos), run_time=1.5)
        self.wait(1)
