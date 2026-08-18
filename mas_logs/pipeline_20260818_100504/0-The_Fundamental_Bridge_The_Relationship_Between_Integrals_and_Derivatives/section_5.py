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
        self.setup_layout("Summary & Synthesis", ["Differentiation breaks motion into rates.", "Integration builds position from rates.", "They form a perfect mathematical cycle."])
        
        # Create visual elements
        func = Text("f(x)", color=BLUE)
        deriv = Text("f'(x)", color=YELLOW)
        
        # Apply positioning adjustments as per feedback
        self.place_in_area(func, 'C2', 'C2', scale_factor=0.85)
        self.place_in_area(deriv, 'C5', 'C5', scale_factor=0.85)
        
        # Arrows
        arrow_diff = Arrow(func.get_right() + RIGHT*0.2, deriv.get_left() + LEFT*0.2, color=WHITE, buff=0)
        arrow_integ = Arrow(deriv.get_bottom() + DOWN*0.2, func.get_bottom() + DOWN*0.2, color=WHITE, buff=0)
        
        diff_label = Text("Derivative", font_size=16, color=YELLOW).next_to(arrow_diff, UP, buff=0.1)
        integ_label = Text("Integral", font_size=16, color=GREEN).next_to(arrow_integ, DOWN, buff=0.1)
        
        cycle = VGroup(func, deriv, arrow_diff, arrow_integ, diff_label, integ_label)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), FadeIn(func), FadeIn(arrow_diff), FadeIn(diff_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN), FadeIn(deriv), FadeIn(arrow_integ), FadeIn(integ_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE), Indicate(cycle))
        self.wait(2)
