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
        lines = [
            "Zooming in reveals linear behavior.",
            "Curves locally behave like lines.",
            "Derivative is a local transformation.",
            "It captures instantaneous curve behavior.",
            "Magnification clarifies local steepness."
        ]
        self.setup_layout("The Transformational View (Zooming In)", lines)
        
        # Define objects
        curve = FunctionGraph(lambda x: 0.5 * x**2 + 0.2 * x, x_range=[-2, 2], color=BLUE)
        point = Dot(color=YELLOW)
        
        # Use assets as instructed
        watermelon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/watermelon.svg")
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")

        # Placement per critics/constraints
        self.place_in_area(curve, 'B1', 'E4', scale_factor=0.65)
        self.place_at_grid(point, 'C3', scale_factor=0.7)
        point.move_to(curve.point_from_proportion(0.6))
        
        # Animate
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(curve.animate.scale(2.0, about_point=point.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        tangent_line = Line(start=LEFT*2, end=RIGHT*2, color=RED).rotate(PI/4).move_to(point.get_center())
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        arrow = Arrow(start=point.get_center(), end=point.get_center()+UP*1.5, color=GREEN)
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        # Place and animate ant crawling on watermelon
        self.place_in_area(watermelon, 'E4', 'F6', scale_factor=0.5)
        self.place_at_grid(ant, 'E4', scale_factor=0.3)
        self.play(FadeIn(watermelon), FadeIn(ant))
        self.play(ant.animate.move_to(self.grid['F6']), run_time=2)
        self.wait(1)
