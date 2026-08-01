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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        # Using lines from TEACHING CONTENT block
        self.setup_layout(
            "The Final Mystery: Square vs. Rectangle", 
            [
                'Rectangles always exist, but what about perfect squares?', 
                'For smooth curves, the answer is a proven yes.', 
                'But for jagged fractals, the mystery remains unsolved.'
            ]
        )

        # --- Object Definitions ---
        # Blue loop and rectangle/square for Line 1
        blue_loop = Ellipse(width=3.5, height=2.5, color="#ADD8E6")
        orange_rect = Rectangle(width=2.5, height=1.5, color="#FFA500")
        # Calculated inscribed square for ellipse (approx side length 2.0)
        yellow_square = Square(side_length=2.0, color="#FFFF00")
        
        # Smooth green shape for Line 2
        # Circle with radius 1.41 to fit square with side 2.0 perfectly
        smooth_green = Circle(radius=1.414, color="#00FF00")
        
        # Jagged red-orange fractal for Line 3
        np.random.seed(42)
        points = []
        num_points = 120
        for i in range(num_points):
            theta = (i / num_points) * 2 * np.pi
            r = 1.3 + 0.15 * np.sin(15 * theta) + 0.1 * np.cos(35 * theta) + 0.05 * np.random.rand()
            points.append([r * np.cos(theta), r * np.sin(theta), 0])
        jagged_fractal = Polygon(*points, color="#FF4500")

        # Question mark asset (Issue 37)
        question_mark = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/question.svg")
        question_mark.set_color(WHITE)

        # Positioning (Issues 50, 51)
        # We pre-position these so they are ready for transforms
        self.place_in_area(blue_loop, "B2", "E6")
        self.place_in_area(orange_rect, "B2", "E6")
        self.place_in_area(yellow_square, "B2", "E6")
        self.place_in_area(smooth_green, "B2", "E6")
        self.place_in_area(jagged_fractal, "B2", "E6")
        
        # Position question mark (Issue 49)
        self.place_at_grid(question_mark, "C4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Rectangles always exist, but what about perfect squares?
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.play(Create(blue_loop))
        self.play(Create(orange_rect))
        self.wait(0.5)
        # Shift rectangle into square
        self.play(ReplacementTransform(orange_rect, yellow_square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # For smooth curves, the answer is a proven yes.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(
            ReplacementTransform(blue_loop, smooth_green),
            yellow_square.animate.set_color("#FFFF00") # Ensure visibility
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # But for jagged fractals, the mystery remains unsolved.
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.play(
            ReplacementTransform(smooth_green, jagged_fractal),
            FadeOut(yellow_square)
        )
        
        # Flash multiple potential yellow squares briefly
        flash_sq1 = Square(side_length=0.4, color="#FFFF00")
        flash_sq2 = Square(side_length=0.3, color="#FFFF00").rotate(PI/4)
        flash_sq3 = Square(side_length=0.5, color="#FFFF00").rotate(-PI/6)
        
        # Position flashes relative to grid
        self.place_at_grid(flash_sq1, "C3", scale_factor=1.0)
        self.place_at_grid(flash_sq2, "D5", scale_factor=1.0)
        self.place_at_grid(flash_sq3, "B4", scale_factor=1.0)
        
        for sq in [flash_sq1, flash_sq2, flash_sq3]:
            self.play(FadeIn(sq, run_time=0.2), FadeOut(sq, run_time=0.2))

        # Question mark fades in over the center
        self.play(FadeIn(question_mark))
        self.wait(2)
