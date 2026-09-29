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
        self.setup_layout("Quantifying Uncertainty: The Entropy Formula", [
            "Entropy measures our average uncertainty about an outcome.",
            "We represent this using a probability tree structure.",
            "The formula calculates bits needed to identify words."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in 'Entropy Formula: H(X) = -Σ p(x) log p(x)'. Color: #FFFFFF.
        formula = MathTex(r"H(X) = -\\sum p(x) \\log_2 p(x)", color=WHITE)
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg", color=WHITE)
        icon.scale(0.5).next_to(formula, LEFT)
        
        group = VGroup(formula, icon)
        # Applying the fix from Issue 24/37
        self.place_in_area(group, "B1", "C5", scale_factor=0.9)
        self.play(FadeIn(group))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Highlight 'p(x)' representing probability of an event. Color: #32CD32.
        p_x = formula.get_parts_by_tex(r"p(x)")
        self.play(Indicate(p_x, color=GREEN))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        # Animate the log function curve descending. Color: #FFD700.
        axes = Axes(x_range=[0, 1, 0.2], y_range=[0, 2, 0.5], x_length=3, y_length=2).add_coordinates()
        curve = axes.plot(lambda x: -np.log2(x + 1e-9) / 2, x_range=[0.05, 1], color=GOLD)
        
        # Applying the fixes from Issue 25/26/37
        self.place_at_grid(axes, "E3", scale_factor=0.7)
        self.place_at_grid(curve, "E3", scale_factor=0.7)
        
        self.play(Create(axes), Create(curve))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
