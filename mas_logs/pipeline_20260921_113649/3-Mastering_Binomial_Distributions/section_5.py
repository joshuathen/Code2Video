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
        self.setup_layout("Real-World Application: Quality Control", [
            "Apply Binomial distribution to quality control.",
            "Calculate k defects in n units.",
            "Example: 1 defect in 50 units."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a factory belt graphic with 50 items.
        factory = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg")
        belt_items = VGroup(*[Circle(radius=0.1, color=BLUE) for _ in range(10)])
        belt_items.arrange(RIGHT, buff=0.1)
        belt_group = VGroup(factory, belt_items).arrange(DOWN)
        self.place_in_area(belt_group, "A1", "C2", scale_factor=0.6)
        self.play(FadeIn(belt_group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Overlay the binomial formula specifically for k=1.
        formula = MathTex(r"P(X=1) = \binom{50}{1} (0.02)^1 (0.98)^{49}")
        self.place_at_grid(formula, "B4", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Show a highlighted 'Defect' probability result bar.
        result = MathTex(r"\approx 0.3716")
        self.place_at_grid(result, "C4", scale_factor=0.9)
        self.play(FadeIn(result))
        self.lecture[2].set_color(ORANGE)
        self.wait(2)
