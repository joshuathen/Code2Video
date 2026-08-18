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
        self.setup_layout("Independence: The 'No Influence' Rule", [
            "Independence means one event doesn't influence another.",
            "Knowing event B provides zero information about A.",
            "Mathematically, P(A|B) must equal P(A)."
        ])
        
        # Elements
        circle_a = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        circle_b = Circle(radius=0.5, color=RED, fill_opacity=0.5)
        
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        die = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg")
        
        label_a = MathTex("A").scale(0.7)
        label_b = MathTex("B").scale(0.7)
        
        # Attach labels
        label_a.next_to(circle_a, UP, buff=0.1)
        label_b.next_to(circle_b, UP, buff=0.1)
        
        # Group assets
        group_a = VGroup(circle_a, label_a, coin.scale(0.5).next_to(circle_a, LEFT, buff=0.1))
        group_b = VGroup(circle_b, label_b, die.scale(0.5).next_to(circle_b, RIGHT, buff=0.1))
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(group_a, "B3")
        self.place_at_grid(group_b, "B6")
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate A moving within constraints
        self.play(group_a.animate.shift(DOWN * 1.5))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        formula = MathTex("P(A|B) = P(A)").set_color("#33FF57")
        self.place_in_area(formula, "E2", "E5", scale_factor=1.2)
        self.lecture[2].set_color("#33FF57")
        self.play(Write(formula))
        self.wait(2)
