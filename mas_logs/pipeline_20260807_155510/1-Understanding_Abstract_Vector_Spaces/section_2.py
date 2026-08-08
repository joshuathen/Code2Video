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
        self.setup_layout("Defining the Arena: The 8 Axioms", [
            "A vector space has two operations.",
            "Addition and scalar multiplication must apply.",
            "Axioms define the game's rules."
        ])
        
        # Load assets
        stadium = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stadium.svg")
        scoreboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scoreboard.svg")
        
        # Create 8 axioms display
        axioms = VGroup(*[Text(f"Axiom {i+1}", font_size=24, color=WHITE) for i in range(8)])
        axioms.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # Place assets and items
        self.place_at_grid(stadium, 'A4', scale_factor=0.5)
        self.place_in_area(axioms, 'C1', 'F3', scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), FadeIn(stadium))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Highlight first 4 axioms
        self.play(*[axiom.animate.set_color("#FF0000") for axiom in axioms[:4]])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"), FadeIn(scoreboard))
        # Highlight remaining 4 axioms
        self.play(*[axiom.animate.set_color("#00FF00") for axiom in axioms[4:]])
        self.place_at_grid(scoreboard, 'E5', scale_factor=0.5)
        self.wait(1)
