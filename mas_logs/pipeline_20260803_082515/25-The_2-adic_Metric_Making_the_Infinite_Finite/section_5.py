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
        lecture_lines = [
            "Let S be the sum 1 plus 2 plus 4.",
            "We can write S as 1 plus 2 times S.",
            "Solving for S gives the result: S equals negative 1.",
            "Infinite positive sums can equal a negative number here.",
            "This mirrors binary overflow in a computer's memory."
        ]
        self.setup_layout("The Sum of All Powers: Calculating -1", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display 'S = 1 + 2 + 4 + 8 +...' #FFFFFF.
        self.lecture[0].set_color(WHITE)
        eq1 = MathTex("S", "=", "1", "+", "2", "+", "4", "+", "8", "+", "\\dots", color=WHITE)
        self.place_at_grid(eq1, "B4", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Group '2 + 4 + 8 +...' and transform it into '2(1 + 2 + 4 +...)' #FFFF00.
        self.lecture[1].set_color(YELLOW)
        eq2 = MathTex("S", "=", "1", "+", "2", "(", "1", "+", "2", "+", "4", "+", "\\dots", ")", color=YELLOW)
        self.place_at_grid(eq2, "C4", scale_factor=1.0)
        
        self.play(TransformMatchingShapes(eq1.copy(), eq2))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Replace the inner sum with 'S' to show 'S = 1 + 2S' and solve for 'S = -1' #00FF00.
        self.lecture[2].set_color(GREEN)
        eq3 = MathTex("S", "=", "1", "+", "2", "S", color=GREEN)
        self.place_at_grid(eq3, "D4", scale_factor=1.0)
        
        self.play(TransformMatchingShapes(eq2.copy(), eq3))
        self.wait(1)
        
        # Intermediate step: S - 2S = 1
        eq4 = MathTex("S", "-", "2", "S", "=", "1", color=GREEN)
        self.place_at_grid(eq4, "E4", scale_factor=1.0)
        self.play(ReplacementTransform(eq3.copy(), eq4))
        self.wait(1)
        
        # Result: S = -1
        eq5 = MathTex("S", "=", "-1", color=GREEN)
        self.place_at_grid(eq5, "F3", scale_factor=1.2)
        self.play(ReplacementTransform(eq4, eq5))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # "Infinite positive sums can equal a negative number here."
        self.lecture[3].set_color(GREEN)
        self.play(Indicate(eq5))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "This mirrors binary overflow in a computer's memory."
        self.lecture[4].set_color(BLUE)
        # Visualizing binary overflow concept: ...1111 = -1
        binary_rep = MathTex("\\dots 1111_2", "=", "-1", color=BLUE)
        self.place_at_grid(binary_rep, "F4", scale_factor=1.2)
        self.play(Write(binary_rep))
        self.wait(3)
