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
            "Conditional probability focuses on a restricted sample space.",
            "Venn diagram shows event B as the new universe.",
            "Knowing B occurred changes our view of event A.",
            "Consider the deck: If Red, is it a Heart?",
            "P(Heart|Red) is thirteen over twenty-six, or point-five."
        ]
        self.setup_layout("Prerequisite Review: Conditional Probability", lecture_lines)
        
        # Assets/Mobjects
        formula = MathTex("P(A|B) = \\frac{P(A \\cap B)}{P(B)}", font_size=36)
        
        # Mocking the Venn Diagram for now as no path provided for VennDiagram_Restricted
        venn_diagram = VGroup(Circle(radius=0.8, color=RED), Circle(radius=0.8, color=BLUE).shift(RIGHT*0.5)).scale(0.5)
        
        cards = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cards.svg").scale(0.5)
        summary_note = Text("P(Heart|Red) = 13/26 = 0.5", font_size=24)
        
        # === Animation for Lecture Line 1 ===
        # Use place_in_area as per Issue 21
        self.place_in_area(formula, 'B2', 'C3', scale_factor=0.6)
        self.play(Write(formula))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Use place_at_grid as per Issue 22
        self.place_at_grid(venn_diagram, 'D3', scale_factor=0.7)
        self.play(Create(venn_diagram))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(cards))
        self.place_at_grid(cards, 'D5', scale_factor=0.7)
        self.lecture[2].set_color("#33FF57")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#3388FF")
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Use place_in_area as per Issue 23
        self.place_in_area(summary_note, 'D4', 'F6', scale_factor=0.5)
        self.play(Write(summary_note))
        self.lecture[4].set_color(YELLOW)
        self.wait(1)
