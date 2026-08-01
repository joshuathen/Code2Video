from manim import *
import numpy as np

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
        # Initialize Scene
        lecture_lines = [
            "Computers process words as unique points in space.",
            "Similar meanings cluster together in this high-dimensional map.",
            "Vector math reveals the deep relationships between concepts."
        ]
        self.setup_layout("Prerequisite: Words as Coordinates (Embeddings)", lecture_lines)

        # Colors
        COLOR_KING = "#3399FF"
        COLOR_QUEEN = "#FF66CC"
        COLOR_APPLE = "#00FF00"
        COLOR_FRUIT = "#CCFFCC"
        COLOR_MATH = "#FFFF00"
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Highlight line
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Display words at random positions
        king = Text("King", font_size=24, color=COLOR_KING)
        queen = Text("Queen", font_size=24, color=COLOR_QUEEN)
        apple = Text("Apple", font_size=24, color=COLOR_APPLE)
        fruit = Text("Fruit", font_size=24, color=COLOR_FRUIT)
        
        self.place_at_grid(king, "A4")
        self.place_at_grid(queen, "D1")
        self.place_at_grid(apple, "F3")
        self.place_at_grid(fruit, "B6")
        
        self.play(FadeIn(king), FadeIn(queen), FadeIn(apple), FadeIn(fruit))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Define dots for clusters
        dot_king = Dot(color=COLOR_KING)
        dot_queen = Dot(color=COLOR_QUEEN)
        dot_apple = Dot(color=COLOR_APPLE)
        dot_fruit = Dot(color=COLOR_FRUIT)
        
        # Transition words into semantic clusters and dots
        # Issue 37: Label King at B1 (dot B2), Label Queen at B5 (dot B4)
        # Issue 38: Label Apple at E3 (dot E4), Label Fruit at E6 (dot E5)
        self.place_at_grid(dot_king, "B2")
        self.place_at_grid(dot_queen, "B4")
        self.place_at_grid(dot_apple, "E4")
        self.place_at_grid(dot_fruit, "E5")
        
        self.play(
            king.animate.move_to(self.grid["B1"]),
            queen.animate.move_to(self.grid["B5"]),
            apple.animate.move_to(self.grid["E3"]),
            fruit.animate.move_to(self.grid["E6"]),
            Create(dot_king), Create(dot_queen), Create(dot_apple), Create(dot_fruit),
            run_time=2
        )
        
        # Show similarity circles/zones
        circle1 = Circle(radius=1.2, color=BLUE, stroke_opacity=0.3).move_to(self.grid["B3"])
        circle2 = Circle(radius=0.8, color=GREEN, stroke_opacity=0.3).move_to(self.grid["E4"])
        self.play(Create(circle1), Create(circle2))
        self.wait(1)
        
        # Cleanup circles and bottom cluster for the math logic
        self.play(FadeOut(circle1), FadeOut(circle2), FadeOut(apple), FadeOut(fruit), FadeOut(dot_apple), FadeOut(dot_fruit))

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Math Text
        math_text = Text("King - Man + Woman = Queen", font_size=24, color=COLOR_MATH)
        self.place_in_area(math_text, "A1", "A6")
        
        # Setup coordinates for vector math
        # Issue 39: Man label at D1 (dot D2), Woman label at D5 (dot D4)
        man = Text("Man", font_size=24, color=WHITE)
        woman = Text("Woman", font_size=24, color=WHITE)
        dot_man = Dot(color=WHITE)
        dot_woman = Dot(color=WHITE)
        
        self.place_at_grid(man, "D1")
        self.place_at_grid(dot_man, "D2")
        self.place_at_grid(woman, "D5")
        self.place_at_grid(dot_woman, "D4")
        
        self.play(Write(math_text))
        self.play(FadeIn(man), Create(dot_man), FadeIn(woman), Create(dot_woman))
        
        # Vector arrows: King - Man = Queen - Woman
        arrow1 = Arrow(start=self.grid["D2"], end=self.grid["B2"], buff=0.1, color=COLOR_MATH)
        arrow2 = Arrow(start=self.grid["D4"], end=self.grid["B4"], buff=0.1, color=COLOR_MATH)
        
        self.play(GrowArrow(arrow1))
        self.play(GrowArrow(arrow2))
        self.wait(2)
