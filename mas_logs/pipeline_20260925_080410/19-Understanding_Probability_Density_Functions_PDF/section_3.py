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
        lecture_lines = ["Probabilities are integrals over a specific interval.", "Area under the curve equals probability of range.", "Use geometry to find likelihoods of specific zones."]
        self.setup_layout("The Geometric Interpretation", lecture_lines)
        
        # Create elements
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 2, 0.5], x_length=4, y_length=3)
        graph = axes.plot(lambda x: 1 if 1 <= x <= 4 else 0.5, color=YELLOW)
        pdf_group = VGroup(axes, graph)
        area = axes.get_area(graph, x_range=[2, 3], color=PURPLE, opacity=0.5)
        formula = MathTex(r"P(2 < X < 3)").set_color(WHITE)

        # Positioning using mandatory methods
        self.place_in_area(pdf_group, 'A3', 'E6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#33FF57")
        self.play(Create(pdf_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.play(Create(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF33")
        self.place_at_grid(formula, 'B4', scale_factor=0.75)
        self.play(Write(formula))
        self.wait(2)
