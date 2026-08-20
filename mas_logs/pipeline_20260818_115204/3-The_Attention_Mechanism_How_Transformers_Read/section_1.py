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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Language is ambiguous and context dependent.",
            "Words change meaning based on neighbors.",
            "We need to look at whole sentences simultaneously."
        ]
        self.setup_layout("The Problem: Contextual Ambiguity", lecture_lines)
        
        # Mobjects for animations
        # Use simple Text; set_color_by_t2c is standard in Manim
        sentence = Text("The animal didn't cross the street because it was too tired", font_size=20, color=WHITE)
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        
        # Setup word groupings
        sentence_group = VGroup(sentence, animal_icon).arrange(DOWN)
        self.place_in_area(sentence_group, 'B1', 'E6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.play(FadeIn(sentence_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#90EE90"))
        
        # Fixing the error: Use set_color_by_t2c correctly on the Text object
        sentence.set_color_by_t2c({"animal": "#FFD700", "it": "#FFD700"})
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFB6C1"))
        
        # Highlight logic - ensure indices are valid for the text used
        line = Line(sentence.get_bottom(), sentence.get_bottom(), color="#00FFFF")
        self.play(Create(line))
        self.wait(1)
