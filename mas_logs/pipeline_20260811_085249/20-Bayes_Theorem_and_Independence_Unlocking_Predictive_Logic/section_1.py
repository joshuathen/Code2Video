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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Warm-up: The Concept of Conditional Probability", 
                          ["Conditional probability focuses on a smaller universe.", 
                           "We shrink the space to event B.", 
                           "Then, calculate the overlap with event A."])
        
        # Setup Venn Diagram
        universe = Rectangle(width=4, height=4, color=WHITE)
        circle_a = Circle(radius=1.2, color="#FF0000", fill_opacity=0.3).shift(LEFT * 0.5)
        circle_b = Circle(radius=1.2, color="#0000FF", fill_opacity=0.3).shift(RIGHT * 0.5)
        
        venn_group = VGroup(universe, circle_a, circle_b)
        self.place_in_area(venn_group, 'B4', 'D6', scale_factor=0.9)

        # Assets
        icon_universe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/universe.svg")
        icon_space = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/space.svg")

        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"P(A|B)", color=WHITE, font_size=40)
        self.place_at_grid(formula, 'B2', scale_factor=1.2)
        self.place_at_grid(icon_universe, 'B1', scale_factor=0.3)
        
        self.play(FadeIn(formula), FadeIn(icon_universe), FadeIn(venn_group))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        # Animate shrinking universe to B
        mask = Rectangle(width=4, height=4, color=BLACK, fill_opacity=0.8).move_to(venn_group)
        self.play(FadeIn(mask))
        self.play(circle_b.animate.set_fill(opacity=0.6))
        self.play(self.lecture[1].animate.set_color("#0000FF"))

        # === Animation for Lecture Line 3 ===
        intersection = Intersection(circle_a, circle_b, color="#800080", fill_opacity=0.8)
        label_int = Text("A and B", color="#FFFF00", font_size=24)
        self.place_at_grid(label_int, 'E5', scale_factor=1.0)
        self.place_at_grid(icon_space, 'E4', scale_factor=0.3)
        
        self.play(FadeIn(intersection), Write(label_int), FadeIn(icon_space))
        self.play(self.lecture[2].animate.set_color("#800080"))
        
        self.wait(2)
