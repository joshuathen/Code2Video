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
            "Model changes with ODEs.",
            "Visualize using direction fields.",
            "Apply initial conditions.",
            "Find the specific path.",
            "ODEs are powerful predictive tools."
        ]
        self.setup_layout("Conclusion & Summary", lecture_lines)
        
        # Representations for recap
        icon1 = Circle(radius=0.3, color=BLUE).set_fill(BLUE, opacity=0.5)
        icon2 = Square(side_length=0.5, color=GREEN).set_fill(GREEN, opacity=0.5)
        icon3 = Triangle().scale(0.3).set_color(YELLOW).set_fill(YELLOW, opacity=0.5)
        icon4 = Star(n=5, outer_radius=0.3, color=RED).set_fill(RED, opacity=0.5)
        icon5 = RoundedRectangle(corner_radius=0.1, height=0.4, width=0.8, color=PURPLE).set_fill(PURPLE, opacity=0.5)
        
        icons = VGroup(icon1, icon2, icon3, icon4, icon5).arrange(DOWN, buff=0.4)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(icon1, 'B4', scale_factor=0.8)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), FadeIn(icon1))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(icon2, 'B3', 'F5', scale_factor=0.6)
        self.play(self.lecture[1].animate.set_color("#FFFFFF"), FadeIn(icon2))

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(icon3, 'C5', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color("#FFFFFF"), FadeIn(icon3))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"), FadeIn(icon4))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"), FadeIn(icon5))
        self.play(Flash(self.lecture[4], color="#FFFF00"))

        # Fade out all elements slowly
        self.play(*[FadeOut(obj) for obj in self.mobjects], run_time=3)
