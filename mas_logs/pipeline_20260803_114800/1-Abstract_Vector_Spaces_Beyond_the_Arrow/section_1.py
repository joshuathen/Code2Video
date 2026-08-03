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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Hook: The Universal Mixer", 
            [
                "Are vectors just arrows in space?",
                "Think of a magic kitchen mixing ingredients.",
                "Fruit like apples and bananas are our vectors.",
                "Scaling or adding fruits creates a new smoothie.",
                "If it's a valid smoothie, we've followed the rules."
            ]
        )

        # Assets Creation (Stylized)
        # Apple: Red circle + small brown stem
        apple_body = Circle(radius=0.4, color="#FF0000", fill_opacity=1)
        apple_stem = Rectangle(width=0.05, height=0.15, color="#8B4513", fill_opacity=1).next_to(apple_body, UP, buff=0)
        apple = VGroup(apple_body, apple_stem)

        # Banana: Yellow curved shape (using Ellipse for simplicity)
        banana = Ellipse(width=0.8, height=0.3, color="#FFFF00", fill_opacity=1).rotate(PI/4)

        # Plus Sign
        plus_sign = MathTex("+", color=WHITE, font_size=48)

        # Smoothie Glass: Light gray trapezoid + rim
        glass_sides = Polygon(
            [-0.3, 0, 0], [0.3, 0, 0], [0.5, 1.0, 0], [-0.5, 1.0, 0],
            color="#E0E0E0", fill_opacity=0.6, stroke_width=2
        )
        smoothie_fill = Polygon(
            [-0.28, 0.05, 0], [0.28, 0.05, 0], [0.45, 0.8, 0], [-0.45, 0.8, 0],
            color="#FFD700", fill_opacity=0.8, stroke_width=0
        )
        smoothie_glass = VGroup(glass_sides, smoothie_fill)

        # === Animation for Lecture Line 1 ===
        # "Are vectors just arrows in space?"
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(apple, "C3", scale_factor=1.0)
        self.place_at_grid(banana, "C5", scale_factor=1.0)
        self.play(FadeIn(apple), FadeIn(banana))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Think of a magic kitchen mixing ingredients."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(plus_sign, "C4")
        self.play(Write(plus_sign))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Fruit like apples and bananas are our vectors."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Apple doubles in size
        self.play(apple.animate.scale(2))
        self.wait(0.5)
        
        # Merge fruit icons into a single group at D4 (future smoothie position)
        self.play(
            apple.animate.move_to(self.grid["D4"]),
            banana.animate.move_to(self.grid["D4"]),
            plus_sign.animate.set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Scaling or adding fruits creates a new smoothie."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.place_at_grid(smoothie_glass, "D4", scale_factor=1.5)
        self.play(
            ReplacementTransform(apple, smoothie_glass),
            FadeOut(banana),
            FadeOut(plus_sign)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "If it's a valid smoothie, we've followed the rules."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        glowing_box = SurroundingRectangle(smoothie_glass, color="#00FF00", buff=0.3)
        glowing_box.set_stroke(width=6)
        
        self.play(Create(glowing_box))
        self.play(Indicate(glowing_box, color="#00FF00", scale_factor=1.1))
        self.wait(2)

        # Final state
        self.lecture[4].set_color(WHITE)
        self.wait(1)
