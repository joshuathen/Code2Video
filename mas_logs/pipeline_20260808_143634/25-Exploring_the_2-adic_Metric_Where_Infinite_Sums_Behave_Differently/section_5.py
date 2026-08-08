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
            "Distance definitions fundamentally reshape mathematical behavior.",
            "Convergence relies on the underlying metric space.",
            "Different metrics reveal new mathematical truths."
        ]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        # Define visual elements
        universe_label = Text("Real Metric", color=BLUE).scale(0.6)
        p_adic_label = Text("2-adic Metric", color=YELLOW).scale(0.6)
        
        standard_line = NumberLine(x_range=[-2, 2], length=4, color=BLUE)
        p_adic_tree = VGroup(*[Circle(radius=0.1, color=YELLOW, fill_opacity=0.5) for _ in range(5)])
        p_adic_tree.arrange(RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(standard_line, 'A2', 'B4', scale_factor=1.0)
        self.place_at_grid(universe_label, 'A2', scale_factor=0.8)
        self.play(Create(standard_line), Write(universe_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(p_adic_tree, 'D2', 'E4', scale_factor=1.0)
        self.place_at_grid(p_adic_label, 'D2', scale_factor=0.8)
        self.play(FadeIn(p_adic_tree), Write(p_adic_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        final_thought = Text("Mathematical Truth is Contextual", color=WHITE, font_size=30)
        self.place_at_grid(final_thought, 'C5', scale_factor=0.9)
        self.play(FadeIn(final_thought))
        self.wait(3)
