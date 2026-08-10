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
        self.setup_layout("Introduction: The Geometric Puzzle", [
            "Place n points on a circle's circumference.",
            "Connect every pair of points with a chord.",
            "How many regions are created inside the circle?",
            "Maximize these regions by choosing point positions.",
            "Think of it like slicing a pizza optimally."
        ])
        
        # Load Pizza Asset
        pizza = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg")
        
        # Circle setup
        circle = Circle(radius=1.5, color="#87CEEB")
        
        # Applying requested placement
        self.place_in_area(circle, "C4", "E6", scale_factor=0.6)
        pizza.scale(0.3).next_to(circle, UP)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        dot1 = Dot(color="#FFFF00")
        dot1.move_to(circle.point_at_angle(PI/2))
        self.play(FadeIn(pizza), Create(circle), FadeIn(dot1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        dot2 = Dot(color="#FFFF00")
        dot2.move_to(circle.point_at_angle(PI/2 + 2*PI/3))
        chord = Line(dot1.get_center(), dot2.get_center(), color="#FFA500")
        self.play(FadeIn(dot2), Create(chord))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        dot3 = Dot(color="#FFFF00")
        dot3.move_to(circle.point_at_angle(PI/2 + 4*PI/3))
        chord2 = Line(dot2.get_center(), dot3.get_center(), color="#FFA500")
        chord3 = Line(dot3.get_center(), dot1.get_center(), color="#FFA500")
        triangle = VGroup(chord, chord2, chord3)
        triangle.set_color("#FF00FF")
        self.play(FadeIn(dot3), Create(chord2), Create(chord3))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.play(FadeIn(pizza)) # Pizza is already there, but keeping to instruction
        self.wait(1)
