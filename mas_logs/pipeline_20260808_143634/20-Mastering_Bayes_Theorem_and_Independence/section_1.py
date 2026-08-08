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
            "Conditional probability updates beliefs with information.",
            "P(A|B) narrows focus to subset B.",
            "Think: If Red, is it a Heart?"
        ]
        self.setup_layout("Prerequisite Review: Conditional Probability", lecture_lines)
        
        # Assets
        card_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cards.svg")
        heart_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg")
        
        # Objects
        box = Rectangle(width=4, height=4, color=WHITE)
        self.place_in_area(box, 'A1', 'C6', scale_factor=0.6)
        
        set_a = Circle(radius=1, color=WHITE, fill_opacity=0.3)
        set_b = Circle(radius=1, color=WHITE, fill_opacity=0.3)
        set_a.move_to(box.get_center() + LEFT * 0.5)
        set_b.move_to(box.get_center() + RIGHT * 0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=0.5)
        self.play(Create(box), Create(set_a), Create(set_b), FadeIn(card_icon.move_to(box.get_center())))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), run_time=0.5)
        intersection = Intersection(set_a, set_b, color="#FFFF00", fill_opacity=0.8)
        self.add(intersection)
        
        label = Text("Joint", font_size=20, color="#FF00FF")
        self.place_at_grid(label, 'B3', scale_factor=0.7)
        self.play(Write(label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"), run_time=0.5)
        self.play(set_b.animate.set_fill(color="#00FFFF", opacity=0.5))
        
        formula = MathTex("P(A|B) = \\frac{P(A \\cap B)}{P(B)}", font_size=30)
        self.place_in_area(formula, 'D2', 'D5', scale_factor=0.8)
        self.play(FadeIn(formula), FadeIn(heart_icon.next_to(formula, RIGHT)))
        self.wait(2)
