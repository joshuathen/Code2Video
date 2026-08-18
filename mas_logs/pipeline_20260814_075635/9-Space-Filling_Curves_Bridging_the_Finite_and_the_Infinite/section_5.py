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
        self.setup_layout("Synthesis & Philosophical Reflection", ["Infinity connects the finite dimensions.", "Dimensions are not entirely rigid.", "The infinite describes our reality."])
        
        # === Animation for Lecture Line 1 ===
        # Display the original 1D line and 2D area side-by-side.
        line = Line(start=LEFT, end=RIGHT, color=BLUE)
        square = Square(side_length=1.5, color=RED)
        
        # Applying layout fixes from Critic
        self.place_at_grid(line, "C3", scale_factor=0.8)
        self.place_at_grid(square, "C4", scale_factor=0.8)
        
        self.play(Create(line), Create(square))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Fade in the phrase "Dimension is not an absolute" (#00FF00).
        phrase = Text("Dimension is not an absolute", font_size=24, color="#00FF00")
        
        # Applying layout fix from Critic
        self.place_in_area(phrase, "D3", "D5", scale_factor=0.65)
        
        self.play(FadeIn(phrase))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        # Show the Peano curve as a symbol of mathematical wonder.
        curve = VGroup()
        points = [
            [-0.75, -0.75, 0], [-0.75, 0.75, 0], [0.75, 0.75, 0], [0.75, -0.75, 0],
            [0, -0.75, 0], [0, 0.75, 0], [-0.75, 0, 0], [0.75, 0, 0]
        ]
        curve.add(Line(points[0], points[1]), Line(points[1], points[2]), Line(points[2], points[3]), Line(points[3], points[4]))
        curve.set_color(YELLOW)
        
        # Applying layout fix from Critic
        self.place_in_area(curve, "B3", "C5", scale_factor=0.7)
        self.play(Write(curve))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        self.wait(2)
