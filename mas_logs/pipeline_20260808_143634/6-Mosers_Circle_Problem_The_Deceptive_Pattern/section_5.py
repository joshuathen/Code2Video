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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: Mathematical Rigor vs. Intuition", 
                          ["Patterns are just a starting point.", 
                           "Always look for the logical proof.", 
                           "Don't let intuition fool you."])
        
        # === Animation for Lecture Line 1 ===
        # Summarize intuition vs proof
        intuition_text = Text("Intuition", color=YELLOW, font_size=36)
        proof_text = Text("Logical Proof", color=GREEN, font_size=36)
        vs_text = Text("vs.", color=WHITE, font_size=30)
        
        group = VGroup(intuition_text, vs_text, proof_text).arrange(RIGHT, buff=0.3)
        self.place_in_area(group, 'A4', 'B6', scale_factor=0.6)
        self.play(Write(group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Final formula
        formula = MathTex(r"R(n) = \frac{n^4 - 6n^3 + 23n^2 - 18n + 24}{24}", color=BLUE)
        self.place_at_grid(formula, 'C4', scale_factor=0.8)
        self.play(FadeIn(formula))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Final emphasis - Using Tex for checkmark to avoid NameError
        check_mark = Tex(r"$\checkmark$", color=RED)
        self.place_at_grid(check_mark, 'D4', scale_factor=1.0)
        self.play(Create(check_mark))
        self.lecture[2].set_color(RED)
        self.wait(2)
