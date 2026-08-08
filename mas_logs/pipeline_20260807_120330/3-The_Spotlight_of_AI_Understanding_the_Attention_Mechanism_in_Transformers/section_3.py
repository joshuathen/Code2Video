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
        # Use lecture lines exactly as provided in storyboard
        # Note: Storyboard has 3 lines, but animation descriptions reference L1-L5 coloring.
        # We map steps 3, 4, and 5 to the final lecture line (L3).
        lecture_lines = [
            "Not all words in a sentence are equally important.",
            "Attention assigns relevance scores between different words.",
            "This \"spotlight\" focuses on the most meaningful connections."
        ]
        self.setup_layout("The Core Concept: Dynamic Weighting", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load assets for animal and street icons (Issue 24)
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg").set_color(WHITE)
        street_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/street.svg").set_color(WHITE)
        
        # Create sentence fragments with assets
        words_p1 = VGroup(
            Text("The", font_size=24),
            animal_icon,
            Text("didn't", font_size=24),
            Text("cross", font_size=24),
            Text("the", font_size=24)
        ).arrange(RIGHT, buff=0.4)
        
        words_p2 = VGroup(
            street_icon,
            Text("because", font_size=24),
            Text("it...", font_size=24)
        ).arrange(RIGHT, buff=0.4)
        
        # Position using place_in_area to avoid overlaps (Issues 37 & 38)
        self.place_in_area(words_p1, 'C2', 'C6', scale_factor=0.7)
        self.place_in_area(words_p2, 'E2', 'E6', scale_factor=0.7)

        self.play(
            FadeIn(words_p1),
            FadeIn(words_p2),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A spotlight (#FFFF00) shines from 'it' to 'animal' and 'street'.
        it_word = words_p2[2]     # "it..."
        animal_word = words_p1[1] # animal icon
        street_word = words_p2[0] # street icon
        
        # Spotlight to animal (upward beam)
        beam_animal = Polygon(
            it_word.get_top(),
            animal_word.get_bottom() + LEFT * 0.3,
            animal_word.get_bottom() + RIGHT * 0.3,
            fill_opacity=0.3, fill_color="#FFFF00", stroke_width=0
        )
        # Spotlight to street (leftward beam)
        beam_street = Polygon(
            it_word.get_left(),
            street_word.get_right() + UP * 0.3,
            street_word.get_right() + DOWN * 0.3,
            fill_opacity=0.3, fill_color="#FFFF00", stroke_width=0
        )

        self.play(
            FadeIn(beam_animal),
            FadeIn(beam_street),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The spotlight on 'animal' intensifies while 'street' fades.
        self.play(
            beam_animal.animate.set_fill(opacity=0.7),
            beam_street.animate.set_fill(opacity=0.05),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation Step 4 (Mapping to Line 3) ===
        # A '0.92' relevance score appears above 'animal' (#00FF00).
        score_animal = Text("0.92", font_size=28, color="#00FF00")
        # Position accurately above the animal icon (Issue 39)
        score_animal.next_to(animal_word, UP, buff=0.2)
        
        self.play(
            FadeIn(score_animal)
        )
        self.wait(1)

        # === Animation Step 5 (Mapping to Line 3) ===
        # A '0.03' relevance score appears above 'street' (#FF0000).
        score_street = Text("0.03", font_size=28, color="#FF0000")
        # Position accurately above the street icon (Issue 39)
        score_street.next_to(street_word, UP, buff=0.2)

        self.play(
            FadeIn(score_street)
        )
        self.wait(2)
