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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the PDF", [
            "PDF defines probability through area under a curve.",
            "Two rules: curve is positive and area is one.",
            "Total area represents hundred percent certainty."
        ])
        
        # Axes
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], x_length=4, y_length=3, axis_config={"include_tip": False})
        self.place_in_area(axes, "B2", "E5", scale_factor=0.55)
        
        # PDF curve (f(x))
        func = lambda x: 0.5 * x * np.exp(-0.5 * x) * 4 # Simplified PDF shape
        curve = axes.plot(func, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve))
        self.lecture[0].set_color("#33A1FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        ineq = MathTex("f(x) \\geq 0", color=WHITE)
        self.place_at_grid(ineq, "A5", scale_factor=0.7)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon, "B1", scale_factor=0.3)
        
        self.play(Write(ineq), FadeIn(icon))
        
        area = axes.get_area(curve, x_range=[0, 4], color="#33A1FF", opacity=0.5)
        self.play(FadeIn(area))
        
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        total_prob = MathTex(r"\int_{-\infty}^{\infty} f(x) dx = 1", font_size=36, color=WHITE)
        self.place_at_grid(total_prob, "E5", scale_factor=0.7)
        self.play(Write(total_prob))
        
        self.lecture[2].set_color("#FF5733")
        self.wait(2)
