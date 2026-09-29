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
            "PDF is a density, not a probability.",
            "Integration calculates probability from density functions.",
            "Total area must always equal one."
        ]
        self.setup_layout("Summary & Key Takeaways", lecture_lines)
        
        # Elements
        pdf_def = Text("f(x) = Density", font_size=32)
        prob_int = MathTex(r"\int_{a}^{b} f(x) \,dx = P(a < X < b)", font_size=32)
        total_prob = MathTex(r"\int_{-\infty}^{\infty} f(x) \,dx = 1", font_size=36)
        
        # Axes for PDF illustration
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1.2, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-x**2), x_range=[-3, 3])
        area = axes.get_area(curve, x_range=[-3, 3], color=BLUE, opacity=0.3)
        pdf_illustration = VGroup(axes, curve, area)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(pdf_def, "B5", scale_factor=0.6)
        self.play(FadeIn(pdf_def))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_in_area(prob_int, "C4", "C6", scale_factor=0.5)
        self.play(FadeIn(prob_int))
        self.lecture[1].set_color("#FFFF33")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(total_prob, "D5", scale_factor=0.7)
        self.place_in_area(pdf_illustration, "E3", "F6", scale_factor=0.4)
        self.play(Create(pdf_illustration), FadeIn(total_prob))
        self.lecture[2].set_color("#33FF57")
        self.wait(2)
