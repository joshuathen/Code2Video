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
        # Initialize Scene
        lecture_lines = [
            "We can find factorials by dividing the next one.",
            "Dividing three factorial by three gives us two factorial.",
            "Dividing two factorial by two leaves us with one.",
            "Now, divide one factorial by one to find zero.",
            "This logical pattern proves that zero factorial is one."
        ]
        self.setup_layout("The Pattern: Working Backwards", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        formula = Text("(n-1)! = n! / n", color="#00FFFF", font_size=32)
        # Fix: Tightened header alignment to A3-A4 (Issue 31)
        self.place_in_area(formula, "A3", "A4", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        fact_3 = Text("3! = 6", color=WHITE, font_size=32)
        # Fix: Center at X3-X4 and scale to 0.8 (Issue 29, 30)
        self.place_in_area(fact_3, "B3", "B4", scale_factor=0.8)
        
        fact_2 = Text("2! = 2", color=WHITE, font_size=32)
        # Fix: Center at X3-X4 and scale to 0.8 (Issue 29, 30)
        self.place_in_area(fact_2, "C3", "C4", scale_factor=0.8)
        
        arrow_1 = Arrow(start=self.grid["B4"], end=self.grid["C4"], color="#FFFFFF", buff=0.1)
        label_1 = Text("÷ 3", color="#FFFFFF", font_size=28)
        label_1.next_to(arrow_1, RIGHT, buff=0.2)
        
        self.play(Write(fact_3))
        self.play(GrowArrow(arrow_1), Write(label_1))
        self.play(Write(fact_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        fact_1 = Text("1! = 1", color=WHITE, font_size=32)
        # Fix: Center at X3-X4 and scale to 0.8 (Issue 29, 30)
        self.place_in_area(fact_1, "D3", "D4", scale_factor=0.8)
        
        arrow_2 = Arrow(start=self.grid["C4"], end=self.grid["D4"], color="#FFFFFF", buff=0.1)
        label_2 = Text("÷ 2", color="#FFFFFF", font_size=28)
        label_2.next_to(arrow_2, RIGHT, buff=0.2)
        
        self.play(GrowArrow(arrow_2), Write(label_2))
        self.play(Write(fact_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        
        fact_0 = Text("0! = 1", color=WHITE, font_size=32)
        # Fix: Center at X3-X4 and scale to 0.8 (Issue 29, 30)
        self.place_in_area(fact_0, "E3", "E4", scale_factor=0.8)
        
        arrow_3 = Arrow(start=self.grid["D4"], end=self.grid["E4"], color="#FFFFFF", buff=0.1)
        label_3 = Text("÷ 1", color="#FFFFFF", font_size=28)
        label_3.next_to(arrow_3, RIGHT, buff=0.2)
        
        self.play(GrowArrow(arrow_3), Write(label_3))
        self.play(Write(fact_0))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        
        gold_box = SurroundingRectangle(fact_0, color="#FFD700", buff=0.2)
        
        self.play(Create(gold_box))
        self.wait(2)
