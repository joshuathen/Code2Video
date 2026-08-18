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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Fundamental Theorem: The Inverse Relationship", 
                          ["Integration reverses the differentiation process.", 
                           "It acts like an undo button.", 
                           "Differentiating an integral restores the function."])
        
        f_prime = MathTex(r"f'(x)", color="#FF5733")
        f_int = MathTex(r"F(x) = \int f(t) dt", color="#33FF57")
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        
        # Load asset
        undo_button = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/button.svg")
        undo_label = Text("Undo Operation", font_size=18, color=WHITE)
        undo_group = VGroup(undo_button, undo_label).arrange(DOWN, buff=0.1)

        # Grouping for VideoCritic layout fixes
        formula_stack = VGroup(f_prime, arrow, f_int).arrange(DOWN, buff=0.5)

        # Positioning elements
        self.place_in_area(formula_stack, 'B3', 'D5', scale_factor=1.1)
        self.place_at_grid(undo_group, 'E4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), Write(f_prime))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"), Write(f_int), GrowArrow(arrow), FadeIn(undo_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33CCFF"), Indicate(f_prime), Indicate(f_int), Flash(undo_button))
        self.wait(2)
