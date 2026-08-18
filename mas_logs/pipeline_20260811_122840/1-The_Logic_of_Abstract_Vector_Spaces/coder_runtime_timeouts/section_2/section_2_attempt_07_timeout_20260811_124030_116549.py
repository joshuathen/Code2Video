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
        self.setup_layout("The Rulebook: The 8 Axioms", ["Eight axioms define the game.", "Space-Bot strictly obeys these rules.", "Violations trigger a logical error."])
        
        # Asset path
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#4682B4")
        
        axiom_cards = Group()
        for i in range(8):
            card = ImageMobject(asset_path)
            # Spread across grid rows B-E, cols 4-5
            row = "BCDE"[i // 2]
            col = "45"[i % 2]
            self.place_at_grid(card, f"{row}{col}", scale_factor=0.2)
            axiom_cards.add(card)
        
        self.play(FadeIn(axiom_cards))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN_C)
        
        for card in axiom_cards:
            self.play(Indicate(card, color=GREEN_B), run_time=0.15)
        
        # Stacked cards
        stack = Group(*[ImageMobject(asset_path).scale(0.2) for _ in range(8)])
        stack.arrange(DOWN, buff=-0.3)
        stack_title = Text("The 8 Rules", font_size=20, color=WHITE)
        rule_book = Group(stack, stack_title).arrange(DOWN, buff=0.2)
        
        # Place rule_book at B4/C5 area
        self.place_in_area(rule_book, 'B4', 'C5', scale_factor=0.8)
        
        self.play(ReplacementTransform(axiom_cards, rule_book))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED_C)
        
        error_sign = Text("ERROR!", font_size=32, color=RED, weight=BOLD)
        self.place_at_grid(error_sign, 'C6', scale_factor=0.5)
        self.play(FadeIn(error_sign, scale=0.5))
        self.play(Indicate(rule_book))
        self.wait(2)
