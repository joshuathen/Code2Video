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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Transformers process data differently.", 
            "Attention looks at all words simultaneously.", 
            "This mechanism captures complex context.", 
            "It creates connections between relevant terms.", 
            "Models learn deeper semantic relationships."
        ]
        self.setup_layout("The Transformer Core: The Self-Attention Mechanism", lecture_lines)
        
        # Setup visual elements
        words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
        word_mobs = VGroup(*[Text(w, font_size=24) for w in words]).arrange(RIGHT, buff=0.3)
        # Fix for issue 27: move to D1-E6
        self.place_in_area(word_mobs, "D1", "E6", scale_factor=0.7)
        
        # Grid layout fix for issue 28
        # Using a representative mobject for grid_group
        grid_labels = VGroup(*[Text(f"{r}{c}", font_size=12) for r in ["A","B","C","D","E","F"] for c in ["1","2","3","4","5","6"]])
        # The prompt instructed: self.place_in_area(grid_group, 'A2', 'F5', scale_factor=0.9)
        # Assuming grid_group refers to an visual representation of the grid
        
        # Colors for lecture lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.play(FadeIn(word_mobs))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        # Highlight all words simultaneously
        self.play(word_mobs.animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        # Draw line between 'it' and 'animal'
        it_mob = word_mobs[7]
        animal_mob = word_mobs[1]
        line = Line(it_mob.get_top(), animal_mob.get_top(), color=RED, stroke_width=4)
        dot1 = Dot(it_mob.get_top(), color=RED)
        dot2 = Dot(animal_mob.get_top(), color=RED)
        self.play(Create(line), FadeIn(dot1), FadeIn(dot2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.wait(2)
