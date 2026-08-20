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
        lecture_lines = [
            "Combinations track unique chord intersections.",
            "Four points define one intersection.",
            "Number of intersections is C(n,4)."
        ]
        self.setup_layout("Prerequisite Concept: Combinatorics", lecture_lines)
        
        # Elements
        formula = MathTex(r"\binom{n}{4} = \frac{n!}{4!(n-4)!}", color="#2ECC71")
        circle = Circle(radius=0.9, color=WHITE)
        points = [Dot(circle.point_at_angle(a)) for a in [0, PI/2, PI, 3*PI/2]]
        
        # Loading asset as per instructions
        chord_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chord.svg", color="#9B59B6")
        
        chord1 = Line(points[0].get_center(), points[2].get_center(), color="#9B59B6")
        chord2 = Line(points[1].get_center(), points[3].get_center(), color="#9B59B6")
        intersection = Dot(color=YELLOW).scale(0.8)
        
        circle_group = VGroup(circle, chord1, chord2, *points, chord_icon)
        label_n4 = Text("C(n,4)", font_size=24, color="#2ECC71")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#3498DB")
        self.place_at_grid(circle, 'B5', scale_factor=0.9) # Fix for issue 26/41
        self.play(FadeIn(circle))
        self.play(FadeIn(VGroup(*points)))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E67E22")
        self.play(Create(chord1), Create(chord2))
        self.play(FadeIn(intersection), FadeIn(chord_icon.scale(0.5).move_to(circle.get_center())))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#2ECC71")
        self.place_at_grid(label_n4, 'D3', scale_factor=0.7) # Fix for issue 27/42
        self.play(Write(label_n4))
        self.place_in_area(formula, 'C3', 'E5', scale_factor=1.0) # Fix for issue 25/40
        self.play(Write(formula))
        self.wait(2)
