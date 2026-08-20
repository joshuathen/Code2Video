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
        self.setup_layout("The Hook: The Constant Ratio", ["Circles come in all sizes.", "Diameter and circumference are related.", "The ratio is always constant."])
        
        # === Animation for Lecture Line 1 ===
        # Draw a large circle in #FFFFFF with radius 2 using Asset: plate.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg", color=WHITE)
        self.place_in_area(circle, 'B2', 'D4', scale_factor=0.5)
        self.play(FadeIn(circle), self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # Highlight circumference in #FF00FF and diameter in #00FFFF.
        circumference = Circle(radius=1.5, color="#FF00FF")
        self.place_at_grid(circumference, 'C5', scale_factor=0.9)
        diameter = Line(start=circumference.get_left(), end=circumference.get_right(), color="#00FFFF")
        
        self.play(
            Create(circumference),
            Create(diameter),
            self.lecture[1].animate.set_color("#00FFFF")
        )

        # === Animation for Lecture Line 3 ===
        # Show the constant ratio π ≈ 3.14159 in #FFFF00 using Asset: coin.svg.
        pi_text = MathTex(r"\\pi \\approx 3.14159", color="#FFFF00")
        self.place_at_grid(pi_text, 'D3', scale_factor=1.2)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        self.place_at_grid(coin, 'E5', scale_factor=0.4)
        
        self.play(
            Write(pi_text),
            FadeIn(coin),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(2)
